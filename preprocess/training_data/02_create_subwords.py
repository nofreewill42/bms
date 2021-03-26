from pathlib import Path
import sentencepiece as sp


if __name__ == '__main__':
    ds_path_str = Path('data_path.txt').read_text()[:-1]
    ds_path = Path(ds_path_str)
    all_texts_path = ds_path/'all_texts.txt'  # input
    subwords_dir_path = ds_path/'subwords'    # output directory
    bpe_num = 2**15                           # "dictionary" size
    subwords_prefix = f'{subwords_dir_path}/bpe_{bpe_num}'  # output

    # Do the job
    subwords_dir_path.mkdir(parents=True, exist_ok=True)
    sp.SentencePieceTrainer.train(input=str(all_texts_path),
                                  model_prefix=subwords_prefix,
                                  vocab_size=bpe_num,
                                  character_coverage=1.0,
                                  model_type='bpe',
                                  )

