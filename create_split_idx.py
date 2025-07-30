import os
import glob
import numpy as np
from sklearn.model_selection import KFold, train_test_split

def create_sklearn_k_fold_splits(filenames, k=10, val_ratio=0.1, test_ratio=0.1):
    '''
    Creates K sets of 'train', 'val', and 'test' data using sklearn's KFold and train_test_split.

    Args:
        filenames (list): A list of all filenames (e.g., 976 filenames).
        k (int): The number of folds/sets to create (e.g., 10 for 10-fold).
                 Each fold will have a different k-th portion as test data.
        val_ratio (float): The target ratio for the validation set (e.g., 0.1 for 10%).
        test_ratio (float): The target ratio for the test set (e.g., 0.1 for 10%).
        random_state (int, optional): Seed for random number generation for reproducibility.

    Returns:
        list: A list of dictionaries, where each dictionary represents a fold
              and contains 'train', 'val', and 'test' lists of filenames.
    '''

    # Convert filenames to a numpy array for easier indexing with sklearn
    orig_indices = np.arange(len(filenames))

    # Initialize KFold
    # shuffle=True is important to randomize the test folds across different runs
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    all_folds_data = []

    # KFold provides (train_indices, test_indices) where test_indices represent 1/k of the data
    for fold_idx, (train_val_indices, test_indices) in enumerate(kf.split(orig_indices)):
        # The 'test_indices' here will be our final test set for this fold
        test_set = orig_indices[test_indices]

        # The 'train_val_indices' represent the remaining (k-1)/k of the data (approx 90% of total)
        # We now need to split this further into train and validation
        train_val_set = orig_indices[train_val_indices]

        # Calculate the `test_size` for `train_test_split` for the validation set.
        val_size_from_remaining = val_ratio / (1.0 - test_ratio)
        
        # Ensure we don't end up with validation larger than remaining
        if val_size_from_remaining >= 1.0:
            val_size_from_remaining = 0.5 # Or raise an error, depending on desired behavior for edge cases

        train_set, val_set = train_test_split(
            train_val_set,
            test_size=val_size_from_remaining,
            random_state=42
        )
        
        all_folds_data.append({
            'train': np.sort(train_set).tolist(),
            'val': np.sort(val_set).tolist(),
            'test': np.sort(test_set).tolist()
        })

    return all_folds_data


datasets = [
            {'name': 'SOF',
             'channel': 'C4'
            },
            {'name': 'DCSM',
             'channel': 'C4-M1'
            },
            {'name': 'HMC',
             'channel': 'EEG_C4-M1'
            },
            {'name': 'MSP',
             'channel': 'C4_M1'
            },
            {'name': 'DOD-H',
             'channel': 'C3_M2'
            },
            {'name': 'DOD-O',
             'channel': 'C4_M1'
            },
            {'name': 'CFS',
             'channel': 'C4'
            },
            {'name': 'ISRUC',
             'channel': 'C4_M1'
            },
            {'name': 'APPLES',
             'channel': 'C4_M1'
            },
            {'name': 'FDCSR',
            'channel':'C4'     
            },
    ]                    
                
for dataset in datasets:
    print(f'Splitting {dataset['name']} dataset...')
    data_root = os.path.join('./dset', dataset['name'], 'npz', dataset['channel']) 
    print(data_root)
    data_fname_list = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(data_root, '*.npz')))]
    
    folds_data = create_sklearn_k_fold_splits(data_fname_list,k=5)
    
    print(folds_data)
    
    np.save(os.path.join('/home/linda/Documents/SleePyCo/split_idx','idx_' + dataset['name'] + '.npy'), folds_data)