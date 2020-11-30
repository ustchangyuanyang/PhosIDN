# PhosIDN
PhosIDN: an integrated deep neural network for improving protein phosphorylation site prediction by combining sequence and protein–protein interaction information Developer: HangyuanYang from Health Informatics Lab, School of Information Science and Technology, University of Science and Technology of China
# Requirement
keras==2.0.0
numpy>=1.8.0
backend==tensorflow
# Related data information need to first load
test data.csv

The input file is an csv file, which includes ptm proteinName, postion, sequences and labels
# Predict for your test data
If you want to use the model to predict your test data, you must prepared the test data as an csv file, the firest column is
proteinName,the seconde col: postion, the third col: sequences

The you can run the predict_PhosIDNSeq.py (using only protein sequence) or predict_PhosIDN.py (using both protein sequence and PPI)

The results is an txt file,like: "Q99440"	"3"	"0.191064" "Q99440"	"13"	"0.10469042" "Q99440"	"19"	"0.099805534" "Q99440"	"33"	"0.16288818" "Q99440"	"42"	"0.5689699" "Q99440"	"60"	"0.13290694" "Q99440"	"64"	"0.10236306" "Q99440"	"80"	"0.14091721" "Q99440"	"82"	"0.034324534"

You can change the corresponding parameters in main function predict_PhosIDNSeq.py or predict_PhosIDN.py to choose to use the model to predict for general or kinase prediction
# Train with your own data
If you want to train your own network,your input file is an csv fie, while contains 4 columns: label, proteinName, postion, sequence label is 1 or 0 represents phoshphorylation and non-phoshphorylation site You can change the corresponding parameters in main function train_phosidnseq.py or train_phosidn.py to choose to use the model to predict for general or kinase prediction
# Contact
Please feel free to contact us if you need any help: yhy1996@mail.ustc.edu.cn
