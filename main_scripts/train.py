if __name__ == '__main__':
    import sys
    import pathlib

    sys.path.append(str(pathlib.Path(__file__).parent.absolute().parent.absolute()))

    import json

    import phononet as pn

    include_files = json.load(open('SankalpRagaDataset.json'))  # import list of song_ids to include
    cd = pn.FullChromaDataset('hindustaniv2.json', '/home/raunak/n_ftt4096__hop_length2048',
                              include_mbids=include_files)  # load dataset of "full song" chromagrams

    train, test = cd.train_test_split(train_size=0.75, random_state=1)  # split training and test sets from "full song" chromagrams
    train = pn.ChromaChunkDataset(train, chunk_size=1500, augmentation=pn.transpose_chromagram)
    test = pn.ChromaChunkDataset(test, chunk_size=1500)

    network = pn.PhonoNetNetwork()  # Can define custom network here
    raga_trainer = pn.RagaTrainer(batch_size=32, gpus=1)
    raga_trainer.fit(network, train, test)
