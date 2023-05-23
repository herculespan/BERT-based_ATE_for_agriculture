SELECT model, 
		ROUND(AVG(test_complete_entities_ratio), 4) test_complete_entities_ratio,
		-- ROUND(AVG(test_partial_entities_ratio), 4)  test_partial_entities_ratio,
		-- ROUND(AVG(test_false_positives_ratio), 4)  test_false_positives_ratio,
		ROUND(AVG(syns_complete_entities_ratio), 4) syns_complete_entities_ratio,
		-- ROUND(AVG(syns_partial_entities_ratio), 4)  syns_partial_entities_ratio,
		-- ROUND(AVG(syns_false_positives_ratio), 4)  syns_false_positives_ratio,
		ROUND(AVG(novel_complete_entities_ratio), 4) novel_complete_entities_ratio
		-- ROUND(AVG(novel_partial_entities_ratio), 4) novel_partial_entities_ratio,
		-- ROUND(AVG(novel_false_positives_ratio), 4) novel_false_positives_ratio
FROM agri_ner_results_0
GROUP BY model
ORDER BY test_complete_entities_ratio DESC;

CREATE TABLE `agri_ner_results_0` (
  `num_experiment` int DEFAULT NULL,
  `model` varchar(100) DEFAULT NULL,
  `best_epoch` int DEFAULT NULL,
  `train_loss` float DEFAULT NULL,
  `train_accuracy` float DEFAULT NULL,
  `val_loss` float DEFAULT NULL,
  `val_accuracy` float DEFAULT NULL,
  `test_accuracy` float DEFAULT NULL,
  `test_f1` float DEFAULT NULL,
  `f1_per_class` text,
  `test_precision` float DEFAULT NULL,
  `precision_per_class` text,
  `test_recall` float DEFAULT NULL,
  `recall_per_class` text,
  `test_complete_entities_ratio` float DEFAULT NULL,
  `test_partial_entities_ratio` float DEFAULT NULL,
  `test_false_positives_ratio` float DEFAULT NULL,
  `syns_complete_entities_ratio` float DEFAULT NULL,
  `syns_partial_entities_ratio` float DEFAULT NULL,
  `syns_false_positives_ratio` float DEFAULT NULL,
  `novel_complete_entities_ratio` float DEFAULT NULL,
  `novel_partial_entities_ratio` float DEFAULT NULL,
  `novel_false_positives_ratio` float DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;