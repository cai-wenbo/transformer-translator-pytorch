from tokenizers.implementations import ByteLevelBPETokenizer
import os




if __name__ == "__main__":
    path_zh = "data/train.zh"
    path_en = "data/train.en"
    tokenizer_zh = ByteLevelBPETokenizer()
    tokenizer_en = ByteLevelBPETokenizer()
    
    save_path_zh = "data/tokenizer_zh"
    save_path_en = "data/tokenizer_en"

    #  training
    tokenizer_zh.train(files=[path_zh], vocab_size=5000, min_frequency=2, special_tokens=[
        "<pad>",
        "<mask>",
        "<unk>",
        "<s>",
        "</s>",
        ])

    tokenizer_en.train(files=[path_en], vocab_size=5000, min_frequency=2, special_tokens=[
        "<pad>",
        "<mask>",
        "<unk>",
        "<s>",
        "</s>",
        ])

    if not os.path.exists(save_path_zh):
        os.makedirs(save_path_zh)
    if not os.path.exists(save_path_en):
        os.makedirs(save_path_en)

    tokenizer_zh.save_model(save_path_zh)
    tokenizer_en.save_model(save_path_en)
    
