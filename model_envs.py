import envs

from transformers import (BertConfig, BertTokenizer, BertModel,
                          RobertaConfig, RobertaTokenizer, RobertaModel,
                          AlbertConfig, AlbertTokenizer, AlbertModel,
                          DebertaConfig, DebertaTokenizer, DebertaModel,
                          DebertaV2Config, DebertaV2Tokenizer, DebertaV2Model)

from transformers import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, DEBERTA_V2_PRETRAINED_CONFIG_ARCHIVE_MAP

############################################################
# Model Related Global Varialbes
############################################################

ALL_MODELS = sum((tuple(conf.keys()) for conf in (BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
                                                  ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
                                                  ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
                                                  DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
                                                  DEBERTA_V2_PRETRAINED_CONFIG_ARCHIVE_MAP)
                  ), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertModel, AlbertTokenizer),
    'deberta': (DebertaConfig, DebertaModel, DebertaTokenizer),
    'deberta-v2': (DebertaV2Config, DebertaV2Model, DebertaV2Tokenizer)
}
