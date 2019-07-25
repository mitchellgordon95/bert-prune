from checkpoint_utils.diff_masks import diff_masks

for task in ['MNLI', 'QQP', 'QNLI', 'SST-2', 'CoLA', 'MRPC', 'RTE']: # 'STS-B'
    print()
    print(task)
    for sparsity in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
        print()
        print(f'SPARSITY {sparsity}')
        for lr in ['2e-5','3e-5','4e-5','5e-5']:
            try:
                print(
                    diff_masks(
                        f'models/{task}/downstream_prune_{int(sparsity*100)}_lr_{lr}',
                        f'models/pretrain/gradual_prune_{int(sparsity*100)}'
                        ),
                    end=' '
                    )
            except:
                print(f'Missing {lr}', end=' ')
