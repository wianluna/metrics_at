def CreateDataLoader(datafolder, data_root, load_size=64, batch_size=1,
                     serial_batches=True, nThreads=4):
    from common.data.bapps.custom_dataset_data_loader import CustomDatasetDataLoader
    import os
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(
        datafolder, dataroot=data_root, load_size=load_size,
        batch_size=batch_size, serial_batches=serial_batches, nThreads=nThreads
    )
    return data_loader
