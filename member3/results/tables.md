# Member 3 - publication tables

data_source: `NF-CSE-CIC-IDS2018-v3 packaged validation (in-domain; not cross-dataset)`

Every row carries `dataset`, `split`, `config_id`, `n_samples`, `data_source` and `notes`.
`status = NOT RUN` means the measurement was blocked; the reason is in the row and in
`results/manifest.json` - a blocked experiment is recorded, never dropped or estimated.

## TABLE1_overall_heldout - Overall held-out performance (current package does not establish cross-dataset generalisation)

| n_samples | n_positive | n_negative | precision | recall | f1 | fpr | fnr | roc_auc | auprc | metric_validity | mean_anomaly_score | median_anomaly_score | unit | decision_column | threshold | threshold_method | dataset | split | n_benign | n_attack | fpr_pct | n_alerts | n_episodes | n_true_alerts | n_missed_episodes | n_false_alerts | detection_rate | alert_precision | mean_detection_delay_steps | median_detection_delay_steps | p95_detection_delay_steps | mean_detection_delay_seconds | median_detection_delay_seconds | false_alert_rate_per_1000_benign_windows | alert_f1 | alert_fpr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1739 | 1015 | 724 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.4522 | 0.5335 | ok | 0.0926 | 0.0924 | sample | is_anomaly | 0.2512 | confidence_aware | dataset2_graph_validation_v2 | test | 724 | 1015 | 0.1381 | 0 | 25 | 0 | 25 | 0 | 0.0000 | 0.0000 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | 0.0000 | 0.0000 | 0.0000 |

## TABLE2_per_attack_family - Per-attack-family performance on the test dataset

| n_samples | n_positive | n_negative | precision | recall | f1 | fpr | fnr | roc_auc | auprc | metric_validity | mean_anomaly_score | median_anomaly_score | n_benign_negative_windows | dataset | attack_family | split | evaluation_contrast | mean_detection_delay_steps | median_detection_delay_steps | mean_detection_delay_seconds | family_alert_rate | threshold | threshold_method |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 136 | 136 | 724 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.3686 | 0.1170 | ok | 0.0822 | 0.0813 | 724 | dataset2_graph_validation_v2 | Bot | test | family vs benign windows | NOT RUN | NOT RUN | NOT RUN | 0.0000 | 0.2512 | confidence_aware |
| 0 | 216 | 724 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.4863 | 0.2150 | ok | NOT RUN | NOT RUN | 724 | dataset2_graph_validation_v2 | Brute_Force_-Web | test | family vs benign windows | NOT RUN | NOT RUN | NOT RUN | 0.0000 | 0.2512 | confidence_aware |
| 0 | 219 | 724 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.4887 | 0.2180 | ok | NOT RUN | NOT RUN | 724 | dataset2_graph_validation_v2 | Brute_Force_-XSS | test | family vs benign windows | NOT RUN | NOT RUN | NOT RUN | 0.0000 | 0.2512 | confidence_aware |
| 23 | 23 | 724 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.5348 | 0.0332 | ok | 0.0965 | 0.1030 | 724 | dataset2_graph_validation_v2 | DDOS_attack-HOIC | test | family vs benign windows | NOT RUN | NOT RUN | NOT RUN | 0.0000 | 0.2512 | confidence_aware |
| 15 | 15 | 724 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.5261 | 0.0213 | ok | 0.0970 | 0.0892 | 724 | dataset2_graph_validation_v2 | DDOS_attack-LOIC-UDP | test | family vs benign windows | NOT RUN | NOT RUN | NOT RUN | 0.0000 | 0.2512 | confidence_aware |
| 34 | 34 | 724 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.4035 | 0.0356 | ok | 0.0849 | 0.0830 | 724 | dataset2_graph_validation_v2 | DDoS_attacks-LOIC-HTTP | test | family vs benign windows | NOT RUN | NOT RUN | NOT RUN | 0.0000 | 0.2512 | confidence_aware |
| 12 | 12 | 724 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.6257 | 0.0324 | ok | 0.1082 | 0.1101 | 724 | dataset2_graph_validation_v2 | DoS_attacks-GoldenEye | test | family vs benign windows | NOT RUN | NOT RUN | NOT RUN | 0.0000 | 0.2512 | confidence_aware |
| 7 | 7 | 724 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.4509 | 0.0094 | ok | 0.0913 | 0.0900 | 724 | dataset2_graph_validation_v2 | DoS_attacks-Hulk | test | family vs benign windows | NOT RUN | NOT RUN | NOT RUN | 0.0000 | 0.2512 | confidence_aware |
| 22 | 22 | 724 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.5078 | 0.0347 | ok | 0.0959 | 0.0943 | 724 | dataset2_graph_validation_v2 | DoS_attacks-SlowHTTPTest | test | family vs benign windows | NOT RUN | NOT RUN | NOT RUN | 0.0000 | 0.2512 | confidence_aware |
| 18 | 18 | 724 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.5780 | 0.0444 | ok | 0.1045 | 0.1084 | 724 | dataset2_graph_validation_v2 | DoS_attacks-Slowloris | test | family vs benign windows | NOT RUN | NOT RUN | NOT RUN | 0.0000 | 0.2512 | confidence_aware |
| 39 | 39 | 724 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.5205 | 0.0528 | ok | 0.0959 | 0.0972 | 724 | dataset2_graph_validation_v2 | FTP-BruteForce | test | family vs benign windows | NOT RUN | NOT RUN | NOT RUN | 0.0000 | 0.2512 | confidence_aware |
| 453 | 453 | 724 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.4193 | 0.3219 | ok | 0.0871 | 0.0866 | 724 | dataset2_graph_validation_v2 | Infilteration | test | family vs benign windows | NOT RUN | NOT RUN | NOT RUN | 0.0000 | 0.2512 | confidence_aware |
| 0 | 128 | 724 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.4971 | 0.1430 | ok | NOT RUN | NOT RUN | 724 | dataset2_graph_validation_v2 | SQL_Injection | test | family vs benign windows | NOT RUN | NOT RUN | NOT RUN | 0.0000 | 0.2512 | confidence_aware |
| 37 | 37 | 724 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.6890 | 0.0790 | ok | 0.1121 | 0.1119 | 724 | dataset2_graph_validation_v2 | SSH-Bruteforce | test | family vs benign windows | NOT RUN | NOT RUN | NOT RUN | 0.0000 | 0.2512 | confidence_aware |

## TABLE3_leave_one_family_out - Leave-one-attack-family-out zero-day results

| protocol | held_out_family | rows_removed_from_permitted_stages | rows_total_for_family | threshold | memory_changed | n_samples | n_positive | n_benign_negative_windows | f1 | precision | recall | fpr | fnr | auprc | roc_auc | family_alert_rate | mean_detection_delay_steps | mean_detection_delay_seconds | dataset | split | evaluation_contrast | tau_delta_vs_baseline | status | rebuild_permitted_stages |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| leave-one-attack-family-out | Bot | 0 | 136 | 0.2512 | 0 | 860 | 136 | 724 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.1170 | 0.3686 | 0.0000 | NOT RUN | NOT RUN | dataset2_graph_validation_v2 | test | held-out family vs benign windows | 0.0000 | ok | True |
| leave-one-attack-family-out | Brute_Force_-Web | 0 | 216 | 0.2512 | 0 | 940 | 216 | 724 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.2150 | 0.4863 | 0.0000 | NOT RUN | NOT RUN | dataset2_graph_validation_v2 | test | held-out family vs benign windows | 0.0000 | ok | True |
| leave-one-attack-family-out | Brute_Force_-XSS | 0 | 219 | 0.2512 | 0 | 943 | 219 | 724 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.2180 | 0.4887 | 0.0000 | NOT RUN | NOT RUN | dataset2_graph_validation_v2 | test | held-out family vs benign windows | 0.0000 | ok | True |
| leave-one-attack-family-out | DDOS_attack-HOIC | 0 | 23 | 0.2512 | 0 | 747 | 23 | 724 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.0332 | 0.5348 | 0.0000 | NOT RUN | NOT RUN | dataset2_graph_validation_v2 | test | held-out family vs benign windows | 0.0000 | ok | True |
| leave-one-attack-family-out | DDOS_attack-LOIC-UDP | 0 | 15 | 0.2512 | 0 | 739 | 15 | 724 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.0213 | 0.5261 | 0.0000 | NOT RUN | NOT RUN | dataset2_graph_validation_v2 | test | held-out family vs benign windows | 0.0000 | ok | True |
| leave-one-attack-family-out | DDoS_attacks-LOIC-HTTP | 0 | 34 | 0.2512 | 0 | 758 | 34 | 724 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.0356 | 0.4035 | 0.0000 | NOT RUN | NOT RUN | dataset2_graph_validation_v2 | test | held-out family vs benign windows | 0.0000 | ok | True |
| leave-one-attack-family-out | DoS_attacks-GoldenEye | 0 | 12 | 0.2512 | 0 | 736 | 12 | 724 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.0324 | 0.6257 | 0.0000 | NOT RUN | NOT RUN | dataset2_graph_validation_v2 | test | held-out family vs benign windows | 0.0000 | ok | True |
| leave-one-attack-family-out | DoS_attacks-Hulk | 0 | 7 | 0.2512 | 0 | 731 | 7 | 724 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.0094 | 0.4509 | 0.0000 | NOT RUN | NOT RUN | dataset2_graph_validation_v2 | test | held-out family vs benign windows | 0.0000 | ok | True |
| leave-one-attack-family-out | DoS_attacks-SlowHTTPTest | 0 | 22 | 0.2512 | 0 | 746 | 22 | 724 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.0347 | 0.5078 | 0.0000 | NOT RUN | NOT RUN | dataset2_graph_validation_v2 | test | held-out family vs benign windows | 0.0000 | ok | True |
| leave-one-attack-family-out | DoS_attacks-Slowloris | 0 | 18 | 0.2512 | 0 | 742 | 18 | 724 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.0444 | 0.5780 | 0.0000 | NOT RUN | NOT RUN | dataset2_graph_validation_v2 | test | held-out family vs benign windows | 0.0000 | ok | True |
| leave-one-attack-family-out | FTP-BruteForce | 0 | 39 | 0.2512 | 0 | 763 | 39 | 724 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.0528 | 0.5205 | 0.0000 | NOT RUN | NOT RUN | dataset2_graph_validation_v2 | test | held-out family vs benign windows | 0.0000 | ok | True |
| leave-one-attack-family-out | Infilteration | 0 | 453 | 0.2512 | 0 | 1177 | 453 | 724 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.3219 | 0.4193 | 0.0000 | NOT RUN | NOT RUN | dataset2_graph_validation_v2 | test | held-out family vs benign windows | 0.0000 | ok | True |
| leave-one-attack-family-out | SQL_Injection | 0 | 128 | 0.2512 | 0 | 852 | 128 | 724 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.1430 | 0.4971 | 0.0000 | NOT RUN | NOT RUN | dataset2_graph_validation_v2 | test | held-out family vs benign windows | 0.0000 | ok | True |
| leave-one-attack-family-out | SSH-Bruteforce | 0 | 37 | 0.2512 | 0 | 761 | 37 | 724 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.0790 | 0.6890 | 0.0000 | NOT RUN | NOT RUN | dataset2_graph_validation_v2 | test | held-out family vs benign windows | 0.0000 | ok | True |

## TABLE4_operational_latency - Operational latency results (per sample, ms)

| stage | stage_description | mean_ms | median_ms | p95_ms | p99_ms | min_ms | max_ms | n_samples | device | batch_size | embedding_dim | k | index_type | memory_size | cpu_count | system | throughput_samples_per_s | note | config_id | data_source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_embedding | Member-2 encode(graph) | 17.4535 | 14.5660 | 32.9080 | 45.6766 | 11.1618 | 63.2859 | 450 | runtime-selected | 1 | 128 | 8 | hnsw | 1024 | 12 | Windows | 4640.7110 |  | cfg-m3-member2-export-v2-graph-xai | NF-CSE-CIC-IDS2018-v3 packaged validation (in-domain; not cross-dataset) |
| B_faiss_retrieval | FAISS k-NN retrieval | 0.1943 | 0.1628 | 0.3294 | 0.4407 | 0.1074 | 0.5135 | 450 | runtime-selected | 1 | 128 | 8 | hnsw | 1024 | 12 | Windows | 4640.7110 |  | cfg-m3-member2-export-v2-graph-xai | NF-CSE-CIC-IDS2018-v3 packaged validation (in-domain; not cross-dataset) |
| C_thresholding | threshold + confidence | 0.0212 | 0.0161 | 0.0377 | 0.0512 | 0.0110 | 0.1101 | 450 | runtime-selected | 1 | 128 | 8 | hnsw | 1024 | 12 | Windows | 4640.7110 |  | cfg-m3-member2-export-v2-graph-xai | NF-CSE-CIC-IDS2018-v3 packaged validation (in-domain; not cross-dataset) |
| D_alert_verification | multi-stage verification (per-window streaming cost) | 3.2028 | 2.2949 | 5.5717 | 6.7294 | 1.8743 | 8.1870 | 450 | runtime-selected | 1 | 128 | 8 | hnsw | 1024 | 12 | Windows | 4640.7110 | one single-window call, i.e. the online/API cost | cfg-m3-member2-export-v2-graph-xai | NF-CSE-CIC-IDS2018-v3 packaged validation (in-domain; not cross-dataset) |
| E_member3_total | Member-3 total scoring | 0.2155 | 0.1794 | 0.3706 | 0.4777 | 0.1191 | 0.5600 | 450 | runtime-selected | 1 | 128 | 8 | hnsw | 1024 | 12 | Windows | 4640.7110 |  | cfg-m3-member2-export-v2-graph-xai | NF-CSE-CIC-IDS2018-v3 packaged validation (in-domain; not cross-dataset) |
| D_alert_verification_batch | multi-stage verification (amortised over one batch) | 0.0369 | 0.0369 | 0.0369 | 0.0369 | 0.0369 | 0.0369 | 150 | runtime-selected | 1 | 128 | 8 | hnsw | 1024 | 12 | Windows | 4640.7110 | one batch pass over all sampled windows, divided by the window count | cfg-m3-member2-export-v2-graph-xai | NF-CSE-CIC-IDS2018-v3 packaged validation (in-domain; not cross-dataset) |




## TABLE8_fidelity_plus - Fidelity+ explanation results

| n_alerts | mean | median | std | ci95_low | ci95_high | frac_score_drops | mean_before | mean_after | masking_method | row_level | dataset | mask_target | top_k | formula | attention_minus_random_mean_pct | n_samples_without_graph | attack_family |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 34.5754 | 34.5754 | NOT RUN | 34.5754 | 34.5754 | 1.0000 | 0.2606 | 0.1705 | attention | method | ALL_TEST_ALERTS | nodes | 3.0000 | 100*(s_before - s_after)/max(|s_before|,eps), s = 1 - max cosine sim to normal memory | -9.3093 | 0.0000 | NOT RUN |
| 1 | 2.4387 | 2.4387 | NOT RUN | 2.4387 | 2.4387 | 1.0000 | 0.2606 | 0.2543 | lowest_attention | method | ALL_TEST_ALERTS | nodes | 3.0000 | 100*(s_before - s_after)/max(|s_before|,eps), s = 1 - max cosine sim to normal memory | -9.3093 | 0.0000 | NOT RUN |
| 1 | 43.8847 | 43.8847 | NOT RUN | 43.8847 | 43.8847 | 1.0000 | 0.2606 | 0.1463 | random | method | ALL_TEST_ALERTS | nodes | 3.0000 | 100*(s_before - s_after)/max(|s_before|,eps), s = 1 - max cosine sim to normal memory | -9.3093 | 0.0000 | NOT RUN |
| 1 | 34.5754 | 34.5754 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | attention | method_x_attack_family | dataset2_graph_validation_v2 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | benign |


## TABLE5_threshold_ablation - Threshold strategy ablation

| strategy | threshold | n_samples | precision | recall | f1 | fpr | fpr_pct | auprc | roc_auc | n_alerts | alert_precision | detection_rate | dataset | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed_mean3sd | 0.2099 | 1739 | 0.0000 | 0.0000 | 0.0000 | 0.0083 | 0.8287 | 0.5335 | 0.4522 | 0 | NOT RUN | 0.0000 | dataset2_graph_validation_v2 | ok |
| fixed_percentile | 0.2441 | 1739 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 0.1381 | 0.5335 | 0.4522 | 0 | NOT RUN | 0.0000 | dataset2_graph_validation_v2 | ok |
| confidence_aware | 0.2512 | 1739 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 0.1381 | 0.5335 | 0.4522 | 0 | NOT RUN | 0.0000 | dataset2_graph_validation_v2 | ok |

## TABLE6_debounce_ablation - Multi-stage alert verification (debounce) ablation

| debounce_steps | n_samples | threshold | auprc | roc_auc | sample_precision | sample_recall | sample_f1 | sample_fpr | alert_precision | alert_recall | alert_f1 | alert_fpr | alert_fpr_pct | n_alerts | detection_rate | mean_detection_delay_steps | median_detection_delay_seconds | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1739 | 0.2512 | 0.5335 | 0.4522 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 0.1381 | 1 | 0.0000 | NOT RUN | NOT RUN | ok |
| 2 | 1739 | 0.2512 | 0.5335 | 0.4522 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | NOT RUN | NOT RUN | ok |
| 3 | 1739 | 0.2512 | 0.5335 | 0.4522 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | NOT RUN | NOT RUN | ok |
| 4 | 1739 | 0.2512 | 0.5335 | 0.4522 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | NOT RUN | NOT RUN | ok |

## TABLE7_memory_adaptation - Adaptive memory / online benign adaptation results

| memory_mode | selection | memory_size_before | memory_size_after | memory_growth | benign_fpr_before_adaptation | benign_fpr_after_adaptation | fpr_reduction | n_updates | adaptation_status | n_rejected_above_threshold | n_rejected_uncertain | online_windows | test_f1 | test_recall | test_precision | test_fpr | model_weights_updated | dataset | n_samples | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| static | kcenter_greedy | 1024 | 1024 | 0 | NOT RUN | NOT RUN | NOT RUN | 0 | not_applied(static_reference_arm) | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | False | dataset2_graph_validation_v2 | 1739 | ok |
| adaptive | kcenter_greedy | 1024 | 1024 | 0 | NOT RUN | NOT RUN | NOT RUN | 0 | no_online_stream | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | False | dataset2_graph_validation_v2 | 1739 | ok |

## TABLE9_ablation_summary - Consolidated Member-3 ablation summary

| arm | variant | status | n_samples | f1 | precision | recall | fpr | auprc | roc_auc | threshold | alert_f1 | alert_precision | alert_fpr | n_alerts | detection_rate | mean_detection_delay_steps | benign_fpr_before_adaptation | benign_fpr_after_adaptation | memory_growth | n_updates | mean_ms_per_query | recall_at_k_vs_bruteforce |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| threshold | fixed_mean3sd | ok | 1739 | 0.0000 | 0.0000 | 0.0000 | 0.0083 | 0.5335 | 0.4522 | 0.2099 | NOT RUN | NOT RUN | NOT RUN | 0.0000 | 0.0000 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| threshold | fixed_percentile | ok | 1739 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 0.5335 | 0.4522 | 0.2441 | NOT RUN | NOT RUN | NOT RUN | 0.0000 | 0.0000 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| threshold | confidence_aware | ok | 1739 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 0.5335 | 0.4522 | 0.2512 | NOT RUN | NOT RUN | NOT RUN | 0.0000 | 0.0000 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| debounce | 1 | ok | 1739 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 0.5335 | 0.4522 | 0.2512 | 0.0000 | 0.0000 | 0.0014 | 1.0000 | 0.0000 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| debounce | 2 | ok | 1739 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 0.5335 | 0.4522 | 0.2512 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| debounce | 3 | ok | 1739 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 0.5335 | 0.4522 | 0.2512 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| debounce | 4 | ok | 1739 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 0.5335 | 0.4522 | 0.2512 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| memory | static | ok | 1739 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | 0.0000 | 0.0000 | NOT RUN | NOT RUN |
| memory | adaptive | ok | 1739 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | 0.0000 | 0.0000 | NOT RUN | NOT RUN |
| retrieval_backend | hnsw | ok | 200 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | 0.0271 | 1.0000 |
| retrieval_backend | flat | ok | 200 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | 0.0204 | 1.0000 |
| retrieval_backend | numpy | ok | 200 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | 0.0794 | 1.0000 |
