#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to read in and store documents in a sqlite database."""

import argparse
import sqlite3
import json
import os
import logging
import importlib.util
import bz2
import pickle
import spacy

from multiprocessing import Pool as ProcessPool
from tqdm import tqdm
import unicodedata

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

nlp = spacy.load("en_core_web_lg", disable=['parser'])

# ------------------------------------------------------------------------------
# Import helper
# ------------------------------------------------------------------------------

PREPROCESS_FN = None


def init(filename):
    global PREPROCESS_FN
    if filename:
        PREPROCESS_FN = import_module(filename).preprocess


def import_module(filename):
    """Import a module given a full path to the file."""
    spec = importlib.util.spec_from_file_location('doc_filter', filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize(text):
    return unicodedata.normalize('NFD', text)


# ------------------------------------------------------------------------------
# Store corpus.
# ------------------------------------------------------------------------------


def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)


def get_contents(filename):
    """Parse the contents of a file. Each line is a JSON encoded document."""
    global PREPROCESS_FN
    documents = []
    with bz2.open(filename, 'rb') as f:
        for line in f:
            # Parse document
            doc = json.loads(line)
            # Maybe preprocess the document with custom function
            if PREPROCESS_FN:
                doc = PREPROCESS_FN(doc)
            # Skip if it is empty or None
            if not doc:
                continue
            # Add the document
            assert len(doc['text']) == len(doc['text_with_links'])
            _text, _text_with_links = pickle.dumps(doc['text']), pickle.dumps(doc['text_with_links'])

            _text_ner = []
            for sent in doc['text']:
                ent_list = [(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in nlp(sent).ents]
                _text_ner.append(ent_list)
            _text_ner_str = pickle.dumps(_text_ner)

            documents.append((normalize(doc['id']), doc['url'], doc['title'], _text, _text_with_links, _text_ner_str, len(doc['text'])))

    return documents


def store_contents(data_path, save_path, preprocess, num_workers=None):
    """Preprocess and store a corpus of documents in sqlite.

    Args:
        data_path: Root path to directory (or directory of directories) of files
          containing json encoded documents (must have `id` and `text` fields).
        save_path: Path to output sqlite db.
        preprocess: Path to file defining a custom `preprocess` function. Takes
          in and outputs a structured doc.
        num_workers: Number of parallel processes to use when reading docs.
    """
    if os.path.isfile(save_path):
        raise RuntimeError('%s already exists! Not overwriting.' % save_path)

    logger.info('Reading into database...')
    conn = sqlite3.connect(save_path)

    conn.execute("PRAGMA synchronous = OFF")

    c = conn.cursor()
    c.execute("CREATE TABLE documents (id PRIMARY KEY, url, title, text, text_with_links, text_ner, sent_num);")
    conn.commit()

    workers = ProcessPool(num_workers, initializer=init, initargs=(preprocess,))
    files = [f for f in iter_files(data_path)]

    total = 0
    count = 0
    pairs = []
    statement = "INSERT INTO documents VALUES (?,?,?,?,?,?,?)"

    with tqdm(total=len(files)) as pbar:
        for pair in tqdm(workers.imap_unordered(get_contents, files)):
            pairs.extend(pair)
            count += len(pair)
            pbar.update()

            if count >= 50000:
                c.executemany(statement, pairs)
                total += count
                count = 0
                pairs = []

        if count > 0:
            c.executemany(statement, pairs)
            total += count
            count = 0
            pairs = []

    logger.info('Read %d docs.' % total)
    logger.info('Committing...')
    conn.commit()
    conn.close()


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='/path/to/data')
    parser.add_argument('save_path', type=str, help='/path/to/saved/db.db')
    parser.add_argument('--preprocess', type=str, default=None,
                        help=('File path to a python module that defines '
                              'a `preprocess` function'))
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    args = parser.parse_args()

    args_dict = vars(args)
    for a in args_dict:
        logger.info('%-28s  %s' % (a, args_dict[a]))

    store_contents(
        args.data_path, args.save_path, args.preprocess, args.num_workers
    )
