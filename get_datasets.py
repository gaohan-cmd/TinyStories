from datasets import load_dataset
from typing import Dict, List
from llama_model import load_model


def get_dataset():
    # 应用全部训练集，约 2.7 M
    # ds_train = load_dataset('noanabeshima/TinyStoriesV2', split='train')
    # 这里可以调整比例，我只用了 10%，约 270 K
    ds_train = load_dataset('noanabeshima/TinyStoriesV2', split='train[:10%]')
    ds_val = load_dataset('noanabeshima/TinyStoriesV2', split='validation')

    # print(ds_train)
    # print(ds_val)
    # # 查看一下数据示例
    # print(ds_train[:2])
    return ds_train, ds_val


# 数据预处理函数
def process_func(examples: Dict[str, List]):
    max_token = 2048
    _, tokenizer = load_model()
    encoded_texts = tokenizer(examples['text'], add_special_tokens=False)
    ####
    # text = 'Hello, world!'
    #
    # tokenizer(text)
    # # {'input_ids': [1, 15043, 29892, 3186, 29991], 'attention_mask': [1, 1, 1, 1, 1]}
    # tokenizer(text, add_special_tokens=False)
    # # {'input_ids': [15043, 29892, 3186, 29991], 'attention_mask': [1, 1, 1, 1]}
    # # 上面多了一个 1，即 tokenizer.bos_token_id，在 LLaMA 中对应的就是 <s>
    #######
    input_ids_list = encoded_texts['input_ids']

    new_input_ids_list, new_attn_mask_list = [], []
    for input_ids in input_ids_list:
        temp = input_ids[-max_token + 1:] + [tokenizer.eos_token_id]
        new_input_ids_list.append(temp)
        new_attn_mask_list.append([1] * len(temp))
    return {
        "input_ids": new_input_ids_list,
        "attention_mask": new_attn_mask_list
    }


# 批量预处理函数
def batch_process_func():
    ds_train, ds_val = get_dataset()
    ds_train = ds_train.shuffle()

    ds_train = ds_train.map(
        process_func,
        batched=True,
        num_proc=8,
        remove_columns=ds_train.column_names,
        desc='Running tokenizer on train_set: '
    )
    ds_val = ds_val.map(
        process_func,
        batched=True,
        num_proc=8,
        remove_columns=ds_val.column_names,
        desc='Running tokenizer on val_set: '
    )
    print(ds_train)
    print(ds_val)
    return ds_train, ds_val


if __name__ == '__main__':
    get_dataset()
    # batch_process_func()
