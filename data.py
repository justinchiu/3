from torchtext import data
from torchtext import datasets
import io

class StructuredLmDataset(data.Dataset):
    def __init__(
        self, path, text_field, newline_eos=True,
        encoding='utf-8', **kwargs
    ):
        """Create a LanguageModelingDataset given a path and a field.
        Arguments:
            path: Path to the data file.
            text_field: The field that will be used for text data.
            newline_eos: Whether to add an <eos> token for every newline in the
                data file. Default: True.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field)]
        text = []
        with io.open(path, encoding=encoding) as f:
            for line in f:
                pline = text_field.preprocess(line)
                import pdb; pdb.set_trace()
                text += pline
                if newline_eos:
                    text.append(u'<eos>')

        examples = [data.Example.fromlist([text], fields)]
        super(StructuredLmDataset, self).__init__(
            examples, fields, **kwargs)


