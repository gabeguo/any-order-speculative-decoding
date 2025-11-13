print("starting imports: packed_dataset.py")
import torch
print("finished imports: packed_dataset.py")

class PackedDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, tokenizer, max_length=512, is_code=False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_code = is_code
        self.content_key = "content" if is_code else "text"

        if self.is_code:
            self.special_character_mappings = {
                "\n": "<cls>",
                "\t": "<sep>",
                " " * 2: "<unk>",
                " " * 3: "<pad>"
            }

        self.dataset_iterator = iter(self.dataset)
        self.current_doc = list()
        self.current_doc_idx = 0

    def __iter__(self):
        while True:
            current_chunk = list()
            try:
                while len(current_chunk) < self.max_length:
                    # load new document
                    if self.current_doc_idx >= len(self.current_doc):
                        curr_text = next(self.dataset_iterator)[self.content_key]
                        if self.is_code:
                            for char, replacement in self.special_character_mappings.items():
                                curr_text = curr_text.replace(char, replacement)
                        self.current_doc = self.tokenizer.encode(curr_text)
                        self.current_doc.append(self.tokenizer.eos_token_id) # add eos to separate documents
                        self.current_doc_idx = 0 # reset index
                    # add as many tokens as we can from this document
                    num_tokens_to_append = min(
                        len(self.current_doc), # use the whole doc if we can
                        self.max_length - len(current_chunk) # get to length
                    )
                    current_chunk.extend(
                        self.current_doc[self.current_doc_idx:self.current_doc_idx + num_tokens_to_append]
                    )
                    # update index in current document (so we can eventually move on to the next document)
                    self.current_doc_idx += num_tokens_to_append
                yield torch.tensor(current_chunk)
            except StopIteration:
                break

if __name__ == "__main__":
    print("starting main: packed_dataset.py")
    import os
    os.environ["HF_HOME"] = "/atlas/u/gabeguo/cache_sub"
    from dotenv import load_dotenv
    load_dotenv()
    from datasets import load_dataset
    from transformers import AutoTokenizer

    train_dataset = load_dataset("bigcode/starcoderdata", data_dir="python", split="train", streaming=True)
    tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")

    packed_dataset = PackedDataset(train_dataset, tokenizer, is_code=True)
    dataloader = torch.utils.data.DataLoader(packed_dataset, batch_size=2, num_workers=1)
    for idx, batch in enumerate(dataloader):
        print(f"\n{idx}\n")
        for i in range(batch.shape[0]):
            print(f"\nChunk {i}:\n")
            decoded = tokenizer.decode(batch[i].tolist())
            for char, replacement in packed_dataset.special_character_mappings.items():
                decoded = decoded.replace(replacement, char)
            print(decoded)
        if idx > 10:
            break
    print("finished main: packed_dataset.py")
