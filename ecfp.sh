#all proteins all rads- rf
python ecfp_exp.py --dataset chembl_dopamine_d2 --model rf --rad 0.5
python ecfp_exp.py --dataset chembl_dopamine_d2 --model rf --rad 1.0
python ecfp_exp.py --dataset chembl_dopamine_d2 --model rf --rad 1.5
python ecfp_exp.py --dataset chembl_factor_xa --model rf --rad 0.5
python ecfp_exp.py --dataset chembl_factor_xa --model rf --rad 1.0
python ecfp_exp.py --dataset chembl_factor_xa --model rf --rad 1.5
python ecfp_exp.py --dataset postera_sars_cov_2_mpro --model rf --rad 0.5
python ecfp_exp.py --dataset postera_sars_cov_2_mpro --model rf --rad 1.0
python ecfp_exp.py --dataset postera_sars_cov_2_mpro --model rf --rad 1.5
#all proteins all rads - knn
python ecfp_exp.py --dataset chembl_factor_xa --model knn --rad 0.5
python ecfp_exp.py --dataset chembl_factor_xa --model knn --rad 1.0
python ecfp_exp.py --dataset chembl_factor_xa --model knn--rad 1.5
python ecfp_exp.py --dataset chembl_dopamine_d2 --model knn --rad 0.5
python ecfp_exp.py --dataset chembl_dopamine_d2 --model knn --rad 1.0
python ecfp_exp.py --dataset chembl_dopamine_d2 --model knn --rad 1.5
python ecfp_exp.py --dataset postera_sars_cov_2_mpro --model knn --rad 0.5
python ecfp_exp.py --dataset postera_sars_cov_2_mpro --model knn --rad 1.0
python ecfp_exp.py --dataset postera_sars_cov_2_mpro --model knn --rad 1.5
#all proteins all rads - mlp
python ecfp_exp.py --dataset chembl_factor_xa --model mlp --rad 0.5
python ecfp_exp.py --dataset chembl_factor_xa --model mlp --rad 1.0
python ecfp_exp.py --dataset chembl_factor_xa --model mlp--rad 1.5
python ecfp_exp.py --dataset chembl_dopamine_d2 --model mlp --rad 0.5
python ecfp_exp.py --dataset chembl_dopamine_d2 --model mlp --rad 1.0
python ecfp_exp.py --dataset chembl_dopamine_d2 --model mlp --rad 1.5
python ecfp_exp.py --dataset postera_sars_cov_2_mpro --model mlp --rad 0.5
python ecfp_exp.py --dataset postera_sars_cov_2_mpro --model mlp --rad 1.0
python ecfp_exp.py --dataset postera_sars_cov_2_mpro --model mlp --rad 1.5