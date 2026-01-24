from engine.wrapper_student import StudentWrapper
from engine.wrapper_teacher import TeacherWrapper
from engine.wrapper_kd_baseline import BaselineKDWrapper
import open_clip
from transformers import AutoTokenizer

def get_model(args):
    if args.wrapper == "Teacher":
        return TeacherWrapper(args)
    elif args.wrapper == "Student":
        return StudentWrapper(args)
    elif args.wrapper == "BaselineKD":
        return BaselineKDWrapper(args)
    else:
        raise NotImplementedError('Model not implemented!')

def get_tokenizer(args):
    if args.bert_type == 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224':
        tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_type, trust_remote_code=True)
    
    return tokenizer