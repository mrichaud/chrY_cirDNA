library(glue)
library(data.table)
library(ggplot2)

chromosomes <- c(as.character(1:22), "X")

calc_kl <- function(p, q, eps = 1e-12) {
  p <- p + eps
  q <- q + eps
  p <- p / sum(p)
  q <- q / sum(q)
  sum(p * log(p / q))
}

# 1. Compute baseline 'others' profile ONCE outside the loop if it aggregates all chromosomes
all_other_files <- unlist(lapply(chromosomes, function(chr) {
  folder <- glue("../Downloads/test_chr_women/chr{chr}/")
  list.files(folder, pattern = "others", full.names = TRUE)
}))

if (length(all_other_files) == 0) {
  stop("No 'others' files found across chromosomes. Check directory path.")
}

col_list_others <- lapply(all_other_files, function(f) fread(f, header = FALSE)[[1]])
profile_mat_others <- do.call(cbind, col_list_others)
mean_chr_others <- as.vector(rowMeans(profile_mat_others))

# 2. Iterate per chromosome with empty checks
results <- list()

for (chr in chromosomes) {
  folder <- glue("../Downloads/test_chr_women/chr{chr}/")
  
  all_size_files <- list.files(folder, pattern = "_size\\.tsv$", full.names = TRUE)
  
  # Exclude any file containing 'others' anywhere before the extension
  all_files <- grep("others", all_size_files, value = TRUE, invert = TRUE)
  
  # Skip chromosome if no files exist (e.g., chrY in female samples)
  if (length(all_files) == 0) {
    message(glue("Warning: No matching files found for chr{chr}. Skipping."))
    next
  }
  
  col_list <- lapply(all_files, function(f) fread(f, header = FALSE)[[1]])
  profile_mat <- do.call(cbind, col_list)
  
  mean_chr <- as.vector(rowMeans(profile_mat))
  
  # Calculate KL divergence on window (using 1-based index 101:221 for 100:220 bp)
  kl_val <- calc_kl(mean_chr[101:221], mean_chr_others[101:221])
  
  results[[chr]] <- data.frame(chromosome = chr, kl = kl_val)
}

# Combine results
sample_data <- rbindlist(results)
sample_data[, group := 1]

# Plot
ggplot(sample_data, aes(x = group, y = log(kl), label = chromosome)) +
  geom_boxplot(fill = "lightblue3", width = 0.3) +
  geom_point() +
  geom_text(check_overlap = TRUE, position = position_jitter(width = 0.15)) +
  theme_classic()