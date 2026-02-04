from zipfile import ZipFile
import synapseclient

def download_and_extract_data(token, train_entity='syn51514132', val_entity='syn51514110',
                              train_extract_path='./BraTS2021_Training_Data',
                              val_extract_path='./BraTS2021_Validation_Data'):
    syn = synapseclient.Synapse()
    syn.login(authToken=token)

    syn_train = syn.get(entity=train_entity, version=1)
    train_data_path = syn_train.path
    with ZipFile(train_data_path) as zObject:
        zObject.extractall(path=train_extract_path)

    syn_val = syn.get(entity=val_entity, version=1)
    val_data_path = syn_val.path
    with ZipFile(val_data_path) as zObject:
        zObject.extractall(path=val_extract_path)

    print('Training data extracted at:', train_extract_path)
    print('Validation data extracted at:', val_extract_path)
    return train_extract_path, val_extract_path