library(tidyverse)
library(ggthemes)
library(latex2exp)
library(mcmcse)

# General plotting parameters 
my_theme <- theme_tufte(base_size = 24) + theme(plot.title = element_text(hjust = 0.5, size = 14, face="bold"), 
                                                plot.subtitle = element_text(hjust = 0.5)) +
  theme(axis.title=element_text(size=24)) + theme(rect = element_rect(fill = "transparent"))
my_colors <- c(
  "#999999", "#E69F00", "#56B4E9", "#009E73",
  "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
BASE_HEIGHT <- 5
BASE_WIDTH <- 7.0

dir_save <- "../../../Results/Paper_figures/Fig6/"
if(!dir.exists(dir_save)) dir.create(dir_save)

dir_save_supp <- str_c(dir_save, "Supp_fig/")
if(!dir.exists(dir_save_supp)) dir.create(dir_save_supp)

data_obs <- read_csv(str_c("../../../Intermediate/Experimental_data/Data_fructose/Fructose_data.csv"), col_types = cols()) %>%
  mutate(fruc = as.factor(fruc), id = as.factor(id))

# Take the best run, 2% case 
path_pvc_kappa_large <- "../../../Intermediate/Multiple_individuals/Mig1_mod_2A/Ram_sampler/Npart100_nsamp40000_corr0.999_exp_id1_run1/Pvc_quantfruc2.0.csv"
path_pvc_diff_large <- "../../../Intermediate/Multiple_individuals/Mig1_mod_2B/Ram_sampler/Npart100_nsamp40000_corr0.999_exp_id4_run1/Pvc_quantfruc2.0.csv"

pvc_kappa <- read_csv(path_pvc_kappa_large, col_types = cols()) %>%
  mutate(data_set = "kappa")
pvc_diff <- read_csv(path_pvc_diff_large, col_types = cols()) %>%
  mutate(data_set = "diff")
pvc_sim <- pvc_kappa %>%
  bind_rows(pvc_diff)

data_obs_sum <- data_obs %>%
  filter(fruc == 2.0) %>%
  group_by(time) %>%
  summarise(median = median(obs), 
            quant95 = quantile(obs, 0.95), 
            quant05 = quantile(obs, 0.05))

#my_col <- c("#D54C25", "#86A8BF")
my_col <- rev(c("#EBD5A3", "#86A8BF"))
range_data <- tibble(x = c(0.0, 18), y = c(0.0, 0.7))
p <- ggplot(pvc_sim, aes(time)) + 
  geom_ribbon(aes(ymin = y1_med_low, ymax = y1_med_upp, fill=data_set, color=data_set), alpha = .4) + 
  geom_ribbon(aes(ymin = y1_qu05_low, ymax = y1_qu05_upp, fill=data_set, color=data_set), alpha = .4) + 
  geom_ribbon(aes(ymin = y1_qu95_low, ymax = y1_qu95_upp, fill=data_set, color=data_set), alpha = .4) +
  geom_line(data=data_obs_sum, mapping=aes(time, median), size = 1.0) + 
  geom_line(data=data_obs_sum, mapping=aes(time, quant95), size = 1.0) + 
  geom_line(data=data_obs_sum, mapping=aes(time, quant05), size = 1.0) +
  scale_color_manual(values = my_col) + 
  scale_fill_manual(values = my_col) + 
  geom_rangeframe(data=range_data, mapping=aes(x, y), size = 1.0) + 
  labs(x = "", y = "") + 
  my_theme + theme(legend.position = "none")

ggsave(str_c(dir_save, "Pvc_large_2.svg"), p, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)

# Plotting 0.05 case 
path_pvc_kappa_large <- "../../../Intermediate/Multiple_individuals/Mig1_mod_2A/Ram_sampler/Npart100_nsamp40000_corr0.999_exp_id1_run1/Pvc_quantfruc0.05.csv"
path_pvc_diff_large <- "../../../Intermediate/Multiple_individuals/Mig1_mod_2B/Ram_sampler/Npart100_nsamp40000_corr0.999_exp_id4_run1/Pvc_quantfruc0.05.csv"

pvc_kappa <- read_csv(path_pvc_kappa_large, col_types = cols()) %>%
  mutate(data_set = "kappa")
pvc_diff <- read_csv(path_pvc_diff_large, col_types = cols()) %>%
  mutate(data_set = "diff")
pvc_sim <- pvc_kappa %>%
  bind_rows(pvc_diff)

data_obs_sum <- data_obs %>%
  filter(fruc == 0.05) %>%
  group_by(time) %>%
  summarise(median = median(obs), 
            quant95 = quantile(obs, 0.95), 
            quant05 = quantile(obs, 0.05))

p <- ggplot(pvc_sim, aes(time)) + 
  geom_ribbon(aes(ymin = y1_med_low, ymax = y1_med_upp, fill=data_set, color=data_set), alpha = .4) + 
  geom_ribbon(aes(ymin = y1_qu05_low, ymax = y1_qu05_upp, fill=data_set, color=data_set), alpha = .4) + 
  geom_ribbon(aes(ymin = y1_qu95_low, ymax = y1_qu95_upp, fill=data_set, color=data_set), alpha = .4) +
  geom_line(data=data_obs_sum, mapping=aes(time, median), size = 1.0) + 
  geom_line(data=data_obs_sum, mapping=aes(time, quant95), size = 1.0) + 
  geom_line(data=data_obs_sum, mapping=aes(time, quant05), size = 1.0) +
  geom_rangeframe(data=range_data, mapping=aes(x, y), size = 1.0) + 
  scale_color_manual(values = my_col) + 
  scale_fill_manual(values = my_col) + 
  labs(x = "", y = "") + 
  my_theme + theme(legend.position = "none")


ggsave(str_c(dir_save_supp, "Pvc_large_005.svg"), p, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)


## Data for small model-structure (network structure 1)
path_pvc_kappa_small <- "../../../Intermediate/Multiple_individuals/Mig1_mod_1A/Ram_sampler/Npart100_nsamp40000_corr0.999_exp_id4_run1/Pvc_quantfruc2.0.csv"
path_pvc_diff_small <- "../../../Intermediate/Multiple_individuals/Mig1_mod_1B/Ram_sampler/Npart100_nsamp40000_corr0.999_exp_id4_run1/Pvc_quantfruc2.0.csv"

pvc_kappa <- read_csv(path_pvc_kappa_small, col_types = cols()) %>%
  mutate(data_set = "kappa")
pvc_diff <- read_csv(path_pvc_diff_small, col_types = cols()) %>%
  mutate(data_set = "diff")
pvc_sim <- pvc_kappa %>%
  bind_rows(pvc_diff) 


data_obs_sum <- data_obs %>%
  filter(fruc == 2.0) %>%
  group_by(time) %>%
  summarise(median = median(obs), 
            quant95 = quantile(obs, 0.95), 
            quant05 = quantile(obs, 0.05))

p <- ggplot(pvc_sim, aes(time)) + 
  geom_ribbon(aes(ymin = y1_med_low, ymax = y1_med_upp, fill=data_set, color=data_set), alpha = .4) + 
  geom_ribbon(aes(ymin = y1_qu05_low, ymax = y1_qu05_upp, fill=data_set, color=data_set), alpha = .4) + 
  geom_ribbon(aes(ymin = y1_qu95_low, ymax = y1_qu95_upp, fill=data_set, color=data_set), alpha = .4) +
  geom_line(data=data_obs_sum, mapping=aes(time, median), size = 1.0) + 
  geom_line(data=data_obs_sum, mapping=aes(time, quant95), size = 1.0) + 
  geom_line(data=data_obs_sum, mapping=aes(time, quant05), size = 1.0) +
  scale_color_manual(values = my_col) + 
  scale_fill_manual(values = my_col) + 
  geom_rangeframe(data=range_data, mapping=aes(x, y), size = 1.0) + 
  labs(x = "", y = "") + 
  my_theme + theme(legend.position = "none")
ggsave(str_c(dir_save, "Pvc_small_2.svg"), p, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)


# Plot the 0.05 case 
path_pvc_kappa_small <- "../../../Intermediate/Multiple_individuals/Mig1_mod_1A/Ram_sampler/Npart100_nsamp40000_corr0.999_exp_id4_run1/Pvc_quantfruc0.05.csv"
path_pvc_diff_small <- "../../../Intermediate/Multiple_individuals/Mig1_mod_1B/Ram_sampler/Npart100_nsamp40000_corr0.999_exp_id4_run1/Pvc_quantfruc0.05.csv"

pvc_kappa <- read_csv(path_pvc_kappa_small, col_types = cols()) %>%
  mutate(data_set = "kappa")
pvc_diff <- read_csv(path_pvc_diff_small, col_types = cols()) %>%
  mutate(data_set = "diff")
pvc_sim <- pvc_kappa %>%
  bind_rows(pvc_diff) 


data_obs_sum <- data_obs %>%
  filter(fruc == 0.05) %>%
  group_by(time) %>%
  summarise(median = median(obs), 
            quant95 = quantile(obs, 0.95), 
            quant05 = quantile(obs, 0.05))

p <- ggplot(pvc_sim, aes(time)) + 
  geom_ribbon(aes(ymin = y1_med_low, ymax = y1_med_upp, fill=data_set, color=data_set), alpha = .4) + 
  geom_ribbon(aes(ymin = y1_qu05_low, ymax = y1_qu05_upp, fill=data_set, color=data_set), alpha = .4) + 
  geom_ribbon(aes(ymin = y1_qu95_low, ymax = y1_qu95_upp, fill=data_set, color=data_set), alpha = .4) +
  geom_line(data=data_obs_sum, mapping=aes(time, median), size = 1.0) + 
  geom_line(data=data_obs_sum, mapping=aes(time, quant95), size = 1.0) + 
  geom_line(data=data_obs_sum, mapping=aes(time, quant05), size = 1.0) +
  scale_color_manual(values = my_col) + 
  scale_fill_manual(values = my_col) + 
  geom_rangeframe(data=range_data, mapping=aes(x, y), size = 1.0) + 
  labs(x = "", y = "") + 
  my_theme + theme(legend.position = "none")
ggsave(str_c(dir_save_supp, "Pvc_small_05.svg"), p, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)


# Pvc no variance case 
path_pvc_2_no_var <- "../../../Intermediate/Multiple_individuals/Mig1_mod_2B_no_var/Ram_sampler/Npart500_nsamp40000_corr0.999_exp_id4_run1/Pvc_quantfruc0.05.csv"

pvc_sim <- read_csv(path_pvc_2_no_var, col_types = cols()) 
data_obs_sum <- data_obs %>%
  filter(fruc == 2.0) %>%
  group_by(time) %>%
  summarise(median = median(obs), 
            quant95 = quantile(obs, 0.95), 
            quant05 = quantile(obs, 0.05))
p <- ggplot(pvc_sim, aes(time)) + 
  geom_ribbon(aes(ymin = y1_med_low, ymax = y1_med_upp), alpha = .4) + 
  geom_ribbon(aes(ymin = y1_qu05_low, ymax = y1_qu05_upp), alpha = .4) + 
  geom_ribbon(aes(ymin = y1_qu95_low, ymax = y1_qu95_upp), alpha = .4) +
  geom_line(data=data_obs_sum, mapping=aes(time, median), size = 1.0) + 
  geom_line(data=data_obs_sum, mapping=aes(time, quant95), size = 1.0) + 
  geom_line(data=data_obs_sum, mapping=aes(time, quant05), size = 1.0) +
  scale_color_manual(values = my_col) + 
  scale_fill_manual(values = my_col) + 
  labs(x = "", y = "") + 
  my_theme + theme(legend.position = "none")


# For the best inference run 
dir_best <- "../../../Intermediate/Multiple_individuals/Mig1_mod_2B_old/Ram_sampler/Npart100_nsamp40000_corr0.999_exp_id4_run1/"

data_scale <- read_csv(str_c(dir_best, "Scale.csv"), col_types = cols()) %>%
  mutate(sample = 1:40000) %>%
  filter(sample > 0)

range_data <- tibble(x = c(0.0, 2.4))
p <- ggplot(data_scale) + 
  geom_density(aes(scale1), linetype = 1, size = 2.0, color = "#86A8BF") + 
  geom_density(aes(scale2), linetype = 2, size = 2.0, color = "#86A8BF") + 
  geom_rangeframe(data=range_data, mapping=aes(x), size = 1.0) + 
  xlim(0, 2.4) + 
  labs(x = "", y = "") + 
  my_theme + theme(text = element_text(size = 30))
ggsave(str_c(dir_save, "Marg_scale1.svg"), p, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)

range_data <- tibble(x = c(0.0, 2.6))
p <- ggplot(data_scale) + 
  geom_density(aes(scale3), linetype = 1, size = 2.0, color = "#86A8BF") + 
  geom_density(aes(scale4), linetype = 2, size = 2.0, color = "#86A8BF") + 
  geom_rangeframe(data=range_data, mapping=aes(x), size = 1.0) + 
  xlim(0, 2.6) + 
  labs(x = "", y = "") + 
  my_theme +  theme(text = element_text(size = 30))
ggsave(str_c(dir_save, "Marg_scale2.svg"), p, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)

data_mean <- read_csv(str_c(dir_best, "Mean.csv"), col_types = cols()) %>%
  mutate(sample = 1:40000) %>%
  filter(sample > 0)
range_data <- tibble(x = c(-1, 3.5))
p <- ggplot(data_mean) + 
  geom_density(aes(mu3), linetype = 1, size = 2.0, color = "#EBD5A3") + 
  geom_density(aes(mu4), linetype = 2, size = 2.0, color = "#EBD5A3") + 
  geom_rangeframe(data=range_data, mapping=aes(x), size = 1.0) + 
  xlim(-1, 3.5) + 
  labs(x = "", y = "") + 
  my_theme +  theme(text = element_text(size = 30))
ggsave(str_c(dir_save, "Marg_mean2.svg"), p, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)

range_data <- tibble(x = c(4.5, 7))
p <- ggplot(data_mean) + 
  geom_density(aes(mu1), linetype = 1, size = 2.0, color = "#EBD5A3") + 
  geom_density(aes(mu2), linetype = 2, size = 2.0, color = "#EBD5A3") + 
  geom_rangeframe(data=range_data, mapping=aes(x), size = 1.0) + 
  xlim(4.5, 7) + 
  labs(x = "", y = "") + 
  my_theme + theme(text = element_text(size = 35))
ggsave(str_c(dir_save_supp, "Marg_mean1.svg"), p, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)

data_corr <- read_csv(str_c(dir_best, "Corr.csv"), col_types = cols()) 
n_corr_param <- 4
n_samples <- 40000
corr_matrix <- matrix(0, nrow = 40000, ncol = sum(seq(from = 1, by = 1, to = n_corr_param-1)))
corr_tibble <- tibble(samples = 1:40000)
for(i in 1:n_corr_param-1){
  
  if(i <= 0) next 
  
  for(j in (i+1):n_corr_param){
    index_choose <- seq(from = i, by = n_corr_param, length.out = n_samples)
    corr_tibble[str_c("c", as.character(i), as.character(j))] <- data_corr[index_choose, j]
  }
}

range_data <- tibble(x = c(0.25,  1.0))
p <- ggplot(corr_tibble, aes(c12)) + 
  geom_density(linetype = 1, size = 2.0, color = "#EBD5A3") + 
  labs(x = "", y = "") + 
  geom_rangeframe(data=range_data, mapping=aes(x), size = 1.0) + 
  xlim(0.25, 1.0) + 
  my_theme + theme(text = element_text(size = 30))
ggsave(str_c(dir_save, "Marg_corr.svg"), p, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)


# To calculate coefficient of variation for log-normal distribution 
calc_cv <- function(mu, sigma) return(sqrt(exp(sigma^2) - 1 ))

# Plot coefficent of variation 
data_coeff_var <- data_mean %>%
  select(-sample) %>%
  bind_cols(data_scale) %>%
  mutate(coeff_var1a = calc_cv(mu1, scale1)) %>%
  mutate(coeff_var1b = calc_cv(mu2, scale2)) %>%
  mutate(coeff_var2a = calc_cv(mu3, scale3)) %>%
  mutate(coeff_var2b = calc_cv(mu4, scale4))

range_data <- tibble(x = c(0.0, 4.0))
p1 <- ggplot(data_coeff_var) + 
  geom_density(aes(coeff_var1a), linetype = 1, size = 2.0, color = "#86A8BF") + 
  geom_density(aes(coeff_var1b), linetype = 2, size = 2.0, color = "#86A8BF") + 
  geom_rangeframe(data=range_data, mapping=aes(x), size = 1.0) + 
  xlim(0, 4) + 
  labs(x = "", y = "") + 
  my_theme + theme(text = element_text(size = 30))

range_data <- tibble(x = c(0.0, 6.0))
p2 <- ggplot(data_coeff_var) + 
  geom_density(aes(coeff_var2a), linetype = 1, size = 2.0, color = "#86A8BF") + 
  geom_density(aes(coeff_var2b), linetype = 2, size = 2.0, color = "#86A8BF") + 
  xlim(0, 6) + 
  geom_rangeframe(data=range_data, mapping=aes(x), size = 1.0) + 
  labs(x = "", y = "") + 
  my_theme + theme(text = element_text(size = 30))

ggsave(str_c(dir_save, "Cv1.svg"), p1, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)
ggsave(str_c(dir_save, "Cv2.svg"), p2, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)

# Creating bivariate plots of inference result
data_pair <- data_mean %>% 
  bind_cols(data_scale) %>%
  select(-starts_with("sample"))

dir_pair1 <- str_c(dir_save, "Pair_plots/svg/")
dir_pair2 <- str_c(dir_save, "Pair_plots/png/")
col_names <-colnames(data_pair)
if(!dir.exists(dir_pair1)) dir.create(dir_pair1, recursive = T)
if(!dir.exists(dir_pair2)) dir.create(dir_pair2, recursive = T)
for(i in 1:(dim(data_pair)[2]-1)){
  for(j in (i+1):dim(data_pair)[2]){
    
    data_pair <- data_pair %>%
      rename("v1" = col_names[i], "v2" = col_names[j])
    
    p <- ggplot(data_pair, aes(v1, v2)) + 
      geom_density2d(color = my_colors[1], size=1.0) + 
      labs(x = "", y = "") + 
      my_theme
    
    name_save1 <- str_c(dir_pair1,  col_names[i], "_", col_names[j], ".svg")
    name_save2 <- str_c(dir_pair2,  col_names[i], "_", col_names[j], ".png")
    ggsave(name_save1, p, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)
    ggsave(name_save2, p, width = BASE_WIDTH, height = BASE_HEIGHT, dpi=300)
    
    colnames(data_pair)[i] <- col_names[i]
    colnames(data_pair)[j] <- col_names[j]
  }
}



# Plot no correlation case 
path_pvc_nor_corr <- str_c("../../../Intermediate/Multiple_individuals/Mig1_mod_2B/Ram_sampler/Npart100_nsamp40000_corr0.999_exp_id4_run1/",
                             "pred/no_corr/Ram_sampler/Pvc_quantfruc2.0.csv")
pvc_no_corr <- read_csv(path_pvc_nor_corr, col_types = cols()) %>%
  mutate(data_set = "diff")

data_obs_sum <- data_obs %>%
  filter(fruc == 2.0) %>%
  group_by(time) %>%
  summarise(median = median(obs), 
            quant95 = quantile(obs, 0.95), 
            quant05 = quantile(obs, 0.05))

my_col <- c("#D54C25", "#86A8BF")
my_col <- rev(c("#EBD5A3", "#86A8BF"))
range_data <- tibble(x = c(0.0, 18), y = c(0.0, 0.7))
p <- ggplot(pvc_diff, aes(time)) + 
  geom_ribbon(aes(ymin = y1_med_low, ymax = y1_med_upp, fill=data_set, color=data_set), alpha = .4) + 
  geom_ribbon(aes(ymin = y1_qu05_low, ymax = y1_qu05_upp, fill=data_set, color=data_set), alpha = .4) + 
  geom_ribbon(aes(ymin = y1_qu95_low, ymax = y1_qu95_upp, fill=data_set, color=data_set), alpha = .4) +
  geom_line(data=data_obs_sum, mapping=aes(time, median), size = 1.0) + 
  geom_line(data=data_obs_sum, mapping=aes(time, quant95), size = 1.0) + 
  geom_line(data=data_obs_sum, mapping=aes(time, quant05), size = 1.0) +
  scale_color_manual(values = my_col) + 
  scale_fill_manual(values = my_col) + 
  geom_rangeframe(data=range_data, mapping=aes(x, y), size = 1.0) + 
  labs(x = "", y = "") + 
  my_theme + theme(legend.position = "none")
ggsave(str_c(dir_save, "No_corr.svg"), p, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)


# Save the remaning parameters to the supp-figure (Fig. S5)
p1 <- ggplot(corr_tibble, aes(c13)) + 
  geom_density(linetype = 1, size = 2.0, color = "#EBD5A3") + 
  labs(x = "", y = "") + 
  geom_rangeframe(size = 1.0) + 
  my_theme + theme(text = element_text(size = 40))

p2 <- ggplot(corr_tibble, aes(c14)) + 
  geom_density(linetype = 1, size = 2.0, color = "#EBD5A3") + 
  labs(x = "", y = "") + 
  geom_rangeframe(size = 1.0) + 
  my_theme + theme(text = element_text(size = 40))

p3 <- ggplot(corr_tibble, aes(c23)) + 
  geom_density(linetype = 1, size = 2.0, color = "#EBD5A3") + 
  labs(x = "", y = "") + 
  geom_rangeframe(size = 1.0) + 
  my_theme + theme(text = element_text(size = 40))

p4 <- ggplot(corr_tibble, aes(c24)) + 
  geom_density(linetype = 1, size = 2.0, color = "#EBD5A3") + 
  labs(x = "", y = "") + 
  geom_rangeframe(size = 1.0) + 
  my_theme + theme(text = element_text(size = 40))

p5 <- ggplot(corr_tibble, aes(c34)) + 
  geom_density(linetype = 1, size = 2.0, color = "#EBD5A3") + 
  labs(x = "", y = "") + 
  geom_rangeframe(size = 1.0) + 
  my_theme + theme(text = element_text(size = 40))

ggsave(str_c(dir_save_supp, "Corr_13.svg"), p1, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)
ggsave(str_c(dir_save_supp, "Corr_14.svg"), p2, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)
ggsave(str_c(dir_save_supp, "Corr_23.svg"), p3, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)
ggsave(str_c(dir_save_supp, "Corr_24.svg"), p4, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)
ggsave(str_c(dir_save_supp, "Corr_34.svg"), p5, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)

p1 <- ggplot(data_mean, aes(mu5)) + 
  geom_density(linetype = 1, size = 2.0, color = "#EBD5A3") + 
  labs(x = "", y = "") +  
  geom_rangeframe(size = 1.0) + 
  my_theme + theme(text = element_text(size = 40))
p2 <- ggplot(data_mean, aes(mu6)) + 
  geom_density(linetype = 1, size = 2.0, color = "#EBD5A3") + 
  labs(x = "", y = "") +  
  geom_rangeframe(size = 1.0) + 
  my_theme + theme(text = element_text(size = 40))

p3 <- ggplot(data_scale, aes(scale5)) + 
  geom_density(linetype = 1, size = 2.0, color = "#EBD5A3") + 
  labs(x = "", y = "") +  
  geom_rangeframe(size = 1.0) + 
  my_theme + theme(text = element_text(size = 40))
p4 <- ggplot(data_scale, aes(scale6)) + 
  geom_density(linetype = 1, size = 2.0, color = "#EBD5A3") + 
  labs(x = "", y = "") +  
  geom_rangeframe(size = 1.0) + 
  my_theme + theme(text = element_text(size = 40))

ggsave(str_c(dir_save_supp, "Mu5.svg"), p1, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)
ggsave(str_c(dir_save_supp, "Mu6.svg"), p2, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)
ggsave(str_c(dir_save_supp, "Scale5.svg"), p3, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)
ggsave(str_c(dir_save_supp, "Scale6.svg"), p4, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)

# Save the kappa-parameters 
data_kappa_sigma <- read_csv(str_c(dir_best, "Kappa_sigma.csv"), col_types = cols()) %>%
  mutate(sample = 1:40000) %>%
  filter(sample > 10000)

p1 <- ggplot(data_kappa_sigma, aes(kappa1)) + 
  geom_density(linetype = 1, size = 2.0, color = "#EBD5A3") + 
  labs(x = "", y = "") +  
  geom_rangeframe(size = 1.0) + 
  my_theme + theme(text = element_text(size = 40))

p2 <- ggplot(data_kappa_sigma, aes(kappa2)) + 
  geom_density(linetype = 1, size = 2.0, color = "#EBD5A3") + 
  labs(x = "", y = "") +  
  geom_rangeframe(size = 1.0) + 
  my_theme + theme(text = element_text(size = 40))
  
p3 <- ggplot(data_kappa_sigma, aes(kappa3)) + 
  geom_density(linetype = 1, size = 2.0, color = "#EBD5A3") + 
  labs(x = "", y = "") +  
  geom_rangeframe(size = 1.0) + 
  my_theme + theme(text = element_text(size = 40))

p4 <- ggplot(data_kappa_sigma, aes(kappa4)) + 
  geom_density(linetype = 1, size = 2.0, color = "#EBD5A3") + 
  labs(x = "", y = "") +  
  geom_rangeframe(size = 1.0) + 
  my_theme + theme(text = element_text(size = 40))

ggsave(str_c(dir_save_supp, "Kappa1.svg"), p1, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)
ggsave(str_c(dir_save_supp, "Kappa2.svg"), p2, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)
ggsave(str_c(dir_save_supp, "Kappa3.svg"), p3, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)
ggsave(str_c(dir_save_supp, "Kappa4.svg"), p4, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)


path_pvc_ssa <- str_c("../../../Intermediate/Multiple_individuals/Mig1_mod_2B/Ram_sampler/Npart100_nsamp40000_corr0.999_exp_id4_run1/",
                           "pred/ssa/Ram_sampler/Pvc_quantfruc2.0.csv")
data_ssa <- read_csv(path_pvc_ssa, col_types = cols())

ggplot(data_ssa, aes(time)) + 
  geom_ribbon(aes(ymin = y1_med_low, ymax = y1_med_upp), alpha = 0.2) + 
  geom_ribbon(aes(ymin = y1_qu05_low, ymax = y1_qu05_upp), alpha = 0.2) + 
  geom_ribbon(aes(ymin = y1_qu95_low, ymax = y1_qu95_upp), alpha = 0.2) + 
  geom_line(data=data_obs_sum, mapping=aes(time, median), size = 1.0) + 
  geom_line(data=data_obs_sum, mapping=aes(time, quant95), size = 1.0) + 
  geom_line(data=data_obs_sum, mapping=aes(time, quant05), size = 1.0) +
  my_theme

# Process inference results for multiple 
dir_res <- "../../../Intermediate/Multiple_individuals/Mig1_mode_2B_rev/Ram_sampler/"
dirs_res <- list.files(dir_res)
dirs_res <- dirs_res[dirs_res != "Pilot_run_data_alt"]
dirs_res <- dirs_res[dirs_res != "Npart100_nsamp40000_corr0.999_exp_id4_run1_pred"]
data_plot_tmp <- tibble()
for(i in 1:length(dirs_res)){
  dir_i <- dirs_res[i]
  
  exp_id <-as.integer(str_match(dir_i, "exp_id(\\d+)_")[2])
  run <- as.integer(str_match(dir_i, "run(\\d+)")[2])
  n_samples <- as.integer(str_match(dir_i, "nsamp(\\d+)_")[2])
  
  data_mean <- read_csv(str_c(dir_res, dir_i, "/Mean.csv"), col_types = cols())
  data_scale <- read_csv(str_c(dir_res, dir_i, "/Scale.csv"), col_types = cols())
  
  data_corr <- read_csv(str_c(dir_res, dir_i, "/Corr.csv"), col_types = cols()) 
  n_corr_param <- 4
  corr_matrix <- matrix(0, nrow = n_samples, ncol = sum(seq(from = 1, by = 1, to = n_corr_param-1)))
  corr_tibble <- tibble(samples = 1:n_samples)
  for(i in 1:n_corr_param-1){
    
    if(i <= 0) next 
    
    for(j in (i+1):n_corr_param){
      index_choose <- seq(from = i, by = n_corr_param, length.out = n_samples)
      corr_tibble[str_c("c", as.character(i), as.character(j))] <- data_corr[index_choose, j]
    }
  }
  
  data_tmp <- data_mean %>%
    bind_cols(corr_tibble, data_scale) %>%
    mutate(run = run, exp_id = exp_id) %>%
    filter(samples > 0)
 
  data_plot_tmp <- data_plot_tmp %>%
    bind_rows(data_tmp)
}

data_plot <- data_plot_tmp %>%
  mutate(exp_id = as.factor(exp_id), run = as.factor(run)) %>%
  filter(exp_id != 6) %>%
  filter(exp_id != 9) %>%
  filter(exp_id != 1) %>%
  filter(exp_id != 10)
  
ggplot(data_plot, aes(c12, color = exp_id)) + 
  geom_density(size = 1.5) + 
  scale_color_manual(values = my_colors) + 
  labs(x = "", y = "") + 
  my_theme + theme(legend.position = "none")
  
data_coeff_var <- data_plot %>%
  select(-samples) %>%
  mutate(coeff_var1a = calc_cv(mu1, scale1)) %>%
  mutate(coeff_var1b = calc_cv(mu2, scale2)) %>%
  mutate(coeff_var2a = calc_cv(mu3, scale3)) %>%
  mutate(coeff_var2b = calc_cv(mu4, scale4))

ggplot(data_coeff_var, aes(color = exp_id)) + 
  geom_density(aes(coeff_var1a), linetype = 1, size = 1.5) + 
  geom_density(aes(coeff_var1b), linetype = 2, size = 1.5) +
  xlim(0.0, 3.0) + 
  labs(x = "", y = "") + 
  scale_color_manual(values = my_colors) + 
  my_theme + theme(legend.position = "none")

ggplot(data_coeff_var, aes(color = exp_id)) + 
  geom_density(aes(coeff_var2a), linetype = 1, size = 1.5) + 
  geom_density(aes(coeff_var2b), linetype = 2, size = 1.5) +
  xlim(0.0, 5.0) + 
  labs(x = "", y = "") + 
  scale_color_manual(values = my_colors) + 
  my_theme + theme(legend.position = "none")

ggplot(data_coeff_var, aes(color = exp_id)) + 
  geom_density(aes(mu3), linetype = 1, size = 1.5) +  
  geom_density(aes(mu4), linetype = 2, size = 1.5) + 
  scale_color_manual(values = my_colors) + 
  labs(x = "", y = "") + 
  my_theme + theme(legend.position = "none")

data_plot_new <- data_plot
  
ggplot(filter(data_plot_new, exp_id == 1), aes(samples, mu1)) + 
  geom_line()
p2 <- ggplot(filter(data_plot_new, exp_id == 2), aes(samples, mu1)) + 
  geom_line()
p3 <- ggplot(filter(data_plot_new, exp_id == 3), aes(samples, mu1)) + 
  geom_line()
ggpubr::ggarrange(p1, p2, p3, ncol = 3)


ggplot(data_plot, aes(samples, mu2, color = exp_id)) +
  geom_smooth(se = F)


ggplot(data_plot, aes(mu1, mu3, color = exp_id)) + 
  geom_density2d()
