python3 main.py --cfg ./replication_configs/reddit-body-gru-0.5.yaml

#for epoch in '1' '5' '10' '20'
#do
#    for lr in '0.001' '0.005' '0.0005' '0.01'
#    do
#        for s in '1' '2' '3'
#        do
#            python3 main.py --cfg ./replication_configs/reddit-body-gru-0.5.yaml --seed $s --TTA_epochs $epoch --TTA_lr_feat $lr
#        done    
#    done
#done
