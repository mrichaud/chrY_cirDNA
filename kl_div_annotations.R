library(glue)
library(data.table)
library(ggplot2)


calc_kl <- function(p, q, eps = 1e-12) {
  p <- p + eps
  q <- q + eps
  p <- p / sum(p)
  q <- q / sum(q)
  sum(p * log(p / q))
}

results <- list()

features <- c("znf", "TxWk", "TxEx", "Tx", "TSS", "ReprPC", "Quies", "PromF", "HET", "GapArtf", "EnhA", "DNase", "BivProm", "Acet")
#features <- c("X-transposed", "X-degenerate", "Ampliconic", "Heterochromatic", "Pseudo-autosomal", "Others")
for (feat in features) {
  print(feat)
  folder <- glue("../Downloads/test_chr_annot/chr22/")
  
  #all_size_files <- list.files(folder, pattern = glue("{feat}_size_hallast\\.tsv$"), full.names = TRUE)
  all_size_files <- list.files(folder, pattern = glue("{feat}_size\\.tsv$"), full.names = TRUE)

  col_list <- lapply(all_size_files, function(f) fread(f, header = FALSE)[[1]])
  profile_mat <- do.call(cbind, col_list)

  mean_chr <- as.vector(rowMeans(profile_mat))
  print(mean_chr)
  
  #all_size_files_other <- list.files(folder, pattern = glue("{feat}_Other_size_hallast\\.tsv$"), full.names = TRUE)
  all_size_files_other <- list.files(folder, pattern = glue("{feat}_Other_size\\.tsv$"), full.names = TRUE)
  
  col_list <- lapply(all_size_files_other, function(f) fread(f, header = FALSE)[[1]])
  profile_mat_other <- do.call(cbind, col_list)
  
  mean_chr_others <- as.vector(rowMeans(profile_mat_other))
  
  
  # Calculate KL divergence on window (using 1-based index 101:221 for 100:220 bp)
  kl_val <- calc_kl(mean_chr[101:221], mean_chr_others[101:221])
  
  results[[feat]] <- data.frame(feature = feat, kl = kl_val)
}

sample_data <- rbindlist(results)
sample_data[, group := 1]

# Plot
ggplot(sample_data, aes(x = group, y = log(kl), label = feature)) +
  geom_boxplot(fill = "lightblue3", width = 0.3) +
  geom_point() +
  geom_text(check_overlap = TRUE, position = position_jitter(width = 0.15)) +
  theme_classic()