# Introduction to Machine Learning with Python

## Chapter 01. Introduction
- dataset.py: p28



## Chapter 02. Supervised Learning
### Some Sample Datasets
in `dataset`:
- forget_dataset.py: p44
- wave_dataset.py: p45
- cancer_dataset.py: p46
- boston_housing_dataset.py: p48

### k-Nearest Neighbors
in `knn`:
- forget_dataset_knn1.py: p49
- forget_dataset_knn3.py: p50
- knn_classifier.py: p51
- knn_classifier_decision_boundary.py: p52
- knn_classifier_cancer.py: p53
- knn_regression1.py: p54
- knn_regression3.py: p55
- knn_regression_prediction.py: p56
- analyze_knr.py: p57

### Linear Models
in `linear`:
- linear_wave.py: p59 
- linear_ols.py: p61
- linear_ols_boston.py: p62
- linear_ridge_boston1.py: p63
- linear_ridge_boston2.py: p66
- linear_lasso.py: p67
- lr_svm.py: p71
- lr_c_parameter.py: p73
- lr_l1_regularization.py: p76
- one_vs_rest_data.py: p78
- one_vs_rest_svm1.py: p79
- one_vs_rest_svm2.py: p80

### Decision Trees
in `decision_trees`:
- dt_20_questions.py: p84
- dt_pre_pruning.py: p89
- dt_not_monotone.py: p92
- dt_regression.py: p94
- dt_and_lr.py: p95

### Random Forest
in `random_forest`:
- random_forest_2_moon.py: p99
- random_forest_cancer.py: p100

### Gradient Boosted Regression Trees
in `gbrt`:
- gbrt_cancer.py: p103

### Kernelized Support Vector Machines
in `k_svm`:
- kernelized_svm1.py: p106
- kernelized_svm2.py: p107
- kernelized_svm3.py: p108
- kernelized_svm4.py: p109
- kernelized_svm5.py: p110
- kernelized_rbf1.py: p112
- kernelized_rbf2.py: p113
- kernelized_rbf3.py: p115
- kernelized_rbf4.py: p116

### Neural Networks
in `nn`:
- mlp_moons_hidden_size_100.py: p122
- mlp_moons_hidden_size_10.py: p123
- mlp_moons_hidden_size_10_10.py: p124
- mlp_moons_hidden_size_10_10_tanh.py: p125
- mlp_moons_alpha.py: p126
- mlp_moons_random.py: p127
- mlp_cancer.py: p128
- mlp_cancer_norm.py: p129

### Uncertainty Estimates from Classifiers
in `estimate`:
- gbrt_decision_function.py: p133
- gbrt_predict_probability.py: p136
- multiclass_estimate.py: p138



## Chapter 03. Unsupervised Learning and Preprocessing
### Preprocessing and Scaling
in `data_transformations`:
- four_scaling.py: p146
- cancer_trans.py: p148
- trans_dataset_same_way.py: p150
- effect_of_preprocess.py: p153

### Dimensionality Reduction, Feature Extraction, and Manifold Learning
in `pca`:
- pca_illustration.py: p154
- per_class_feature_histogram.py: p156
- pca_of_cancer.py: p158
- pca_of_cancer_heat_map.py: p158
- eigenface.py: p161

in `nmf`:
- nmf_to_synthetic_data.py: p170
- reconstruct_face.py: p172
- nmf_15_components.py: p173
- nmf_3rd_component.py: p174
- nmf_7rd_component.py: p174
- signal_source.py: p175
- decomposition_signal_source.py: p176

in `tsne`:
- digits.py: p178
- pca_digits.py: p179
- tsne_digits.py: p180

in `kmeans`:
- clustering_step.py: p182
- clustering_boundary.py: p182
- kmeans.py: p184
- kmeans_2_vs_5.py: p186
- kmeans_failure.py: p187
- kmeans_nonspherical.py: p188
- kmeans_tow_moons.py: p189
- kmeans_pca_nmf.py: p191
- kmeans_cover.py: p194

in `agglomerative`:
- agglomerative_illustration.py: p196
- agglomerative_blob.py: p197
- hierarchical_cluster.py: p198
- dendrogram.py: p199

in `dbscan`:
- dbscan_illustration.py: p202
- dbscan_moons.py: p204

in `compare_algorithms`:
- ari.py: p206
- accuracy.py: p207
- silhouette.py: p207



## Chapter 04. Representing Data and Engineering Features
in 'categorical_variables':
- dummy_variables.py: p228
- dummy_number_variables.py: p232
- lr_vs_dtr.py: p234
- lr_vs_dtr_bins.py: p235
- lr_bins_single_slope.py: p238
- lr_bins_separate_slope.py: p239
- lr_polynomial.py: p241
- lr_polynomial_vs_svr.py: p243
- lr_polynomial_boston.py: p244
- nonlinear.py: p246
- nonlinear_transformation.py: p248
- avona.py: p251
- select_from_model.py: p253
- rfe.py: p254

in `expert_knowledge`:
- citibike_data.py: p256
- citibike_predict_all.py: p258
- citibike_predict_hour.py: p260
- citibike_predict_hour_week.py: p260
- citibike_predict_hour_week_lr.py: p261
- citibike_predict_hour_week_onehot.py: p261
- citibike_predict_hour_week_poly_transformer.py: p262



## Chapter 05. Model Evaluation and Improvement
- lr_score.py: p265

in `cross_validation`:
- diagram.py: p266
- cv_default.py：p267
- ksplit.py: p270
- leave_one_out.py: p271
- shuffle_split.py: p272
- group_kfold.py: p273

in `grid_search`:
- simple_gs.py: p275
- simple_rebuild_gs.py: p276
- simple_cv_gs.py: p278
- cv_selection.py: p278
- gs_overview.py: p279
- gridsearchcv.py: p279
- misspecified_gs_overview.py: p284
- gridsearchcv_kernel.py: p285
- nested_cv.py: p287

in `evaluation`:
- imbalanced_dataset.py: p292
- precision_recall_curve_svc.py: p303
- precision_recall_curve_rfc.py: p305



## Chapter 06. Algorithm Chains and Pipelines
- preprocessing.py: p320
- pipeline.py: p322
- create_pl.py: p327
- boston_pl.py: p331










