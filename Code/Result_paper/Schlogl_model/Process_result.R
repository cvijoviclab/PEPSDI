library(tidyverse)
library(ggthemes)
library(latex2exp)

# General plotting parameters 
my_theme <- theme_tufte(base_size = 24) + theme(plot.title = element_text(hjust = 0.5, size = 14, face="bold"), 
                                                plot.subtitle = element_text(hjust = 0.5)) +
  theme(axis.title=element_text(size=24)) + theme(rect = element_rect(fill = "transparent"))
my_colors <- c(
  "#999999", "#E69F00", "#56B4E9", "#009E73",
  "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
BASE_HEIGHT <- 5
BASE_WIDTH <- 7.0


PriorDist <- setClass("PriorDist", slots = list(dist="character", param="numeric"))

dir_save <- "../../Results/Paper_figures/Fig3/"
if(!dir.exists(dir_save)) dir.create(dir_save)


dir_res <- "../../Intermediate/Multiple_individuals/Schlogl_mult_ind_alt/Ram_sampler/Npart100_nsamp500000_corr0.999_exp_id1_run1/"

data_mean <- read_csv(str_c(dir_res, "Mean.csv"), col_types = cols()) %>%
  mutate(sample = 1:500000) %>%
  filter(sample > 100000 * 0.2)
n_sample <- 500000

p1 <- ggplot(data_mean, aes(mu1)) + 
  geom_density(size = 2.0, color = my_colors[8]) + 
  geom_rangeframe(size = 2.0, color = "black", linetype = 1) + 
  geom_vline(xintercept = 7.2, size = 2.0) + 
  labs(x = "", y = "") + 
  my_theme + theme(legend.position = "none")
ggsave(str_c(dir_save, "Mu1.svg"), p1, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)


data_scale <- read_csv(str_c(dir_res, "Scale.csv"), col_types = cols()) %>%
  mutate(sample = 1:n_sample) %>%
  filter(sample > 100000 * 0.2)

p1 <- ggplot(data_scale, aes(scale1)) + 
  geom_density(size = 2.0, color = my_colors[8]) + 
  geom_rangeframe(size = 2.0, color = "black", linetype = 1) + 
  geom_vline(xintercept = 0.1, size = 2.0) + 
  labs(x = "", y = "") + 
  my_theme + theme(legend.position = "none")
ggsave(str_c(dir_save, "Scale1.svg"), p1, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)

data_kappa_sigma <- read_csv(str_c(dir_res, "Kappa_sigma.csv"), col_types = cols()) %>%
  mutate(sample = 1:500000) %>%
  filter(sample > 500000 * 0.2)

p1 <- ggplot(data_kappa_sigma, aes(kappa1)) + 
  geom_density(size = 2.0, color = my_colors[8]) + 
  geom_rangeframe(size = 2.0, color = "black", linetype = 1) + 
  geom_vline(xintercept = -2.17, size = 2.0) + 
  labs(x = "", y = "") + 
  my_theme + theme(legend.position = "none")
p2 <- ggplot(data_kappa_sigma, aes(kappa2)) + 
  geom_density(size = 2.0, color = my_colors[8]) + 
  geom_rangeframe(size = 2.0, color = "black", linetype = 1) + 
  geom_vline(xintercept = -8.73, size = 2.0) + 
  labs(x = "", y = "") + 
  my_theme + theme(legend.position = "none")
p3 <- ggplot(data_kappa_sigma, aes(sigma1)) + 
  geom_density(size = 2.0, color = my_colors[8]) + 
  geom_rangeframe(size = 2.0, color = "black", linetype = 1) + 
  geom_vline(xintercept = 2.0, size = 2.0) + 
  labs(x = "", y = "") + 
  my_theme + theme(legend.position = "none")


ggsave(str_c(dir_save, "Kappa1.svg"), p1, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)
ggsave(str_c(dir_save, "Kappa2.svg"), p2, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)


path_data <-  "../../Intermediate/Simulated_data/SSA/Multiple_ind/schlogl/schlogl.csv"
data_obs <- read_csv(path_data, col_types = cols(id = col_factor()))
data_rand <- data_obs %>%
  filter(id %in% c(1, 5, 31))

data_frame <- tibble(x = c(0, 50.0), y = c(0, 550))
p <- ggplot(data_obs, aes(time, obs)) + 
  geom_line(aes(group = id), color = my_colors[1], size = 0.1) +
  geom_line(data=data_rand, mapping = aes(time, obs, group = id), color = "#444444", size=0.9) +
  geom_rangeframe(size = 1.0) + 
  labs(x = "", y = "") + 
  my_theme
ggsave(str_c(dir_save, "Data_obs.svg"), p, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)
      

data_pvc <- read_csv(str_c(dir_res, "Sol_mat.csv"), col_types = cols())
data_param <- read_csv(str_c(dir_res, "Pvc_ind_param.csv"), col_types = cols())
t_vec <- seq(from = 0.0, to = 50.0, length.out = dim(data_pvc)[2])

data_sim <- parallel::mclapply(1:dim(data_pvc)[1], function(i){
    
  if(data_param$c1[i] > 15) return(tibble())
  tibble_tmp <- tibble(time = t_vec, y = as.numeric(data_pvc[i, ]), id = i, param = data_param$c1[i])
  return(tibble_tmp)}, mc.cores = 6)

data_sim <- do.call(rbind, data_sim)
data_sim_plot <- data_sim %>%
  mutate(id = as.factor(id)) %>%
  mutate(label = case_when(
    param < 7.15 ~ "1", 
    TRUE ~ "2", 
    param > 7.225 ~ "3", 
    TRUE ~ "2"))

data_sum_plot <- data_sim_plot %>%
  group_by(label, time) %>%
  summarise(median = median(y), 
            quant90 = quantile(y, 0.90), 
            quant10 = quantile(y, 0.10)) %>%
  filter(time > 0.1)

col_val <- c("#41b6c4", "#225ea8")
data_frame <- tibble(x = c(0, 50.0), y = c(50, 550), label = c("1", "1"))
p <- ggplot(data_sum_plot, aes(time, median, color = label, fill = label)) + 
  geom_line(size = 1.5) + 
  geom_ribbon(aes(ymin = quant10, ymax = quant90), alpha = 0.2) + 
  scale_color_manual(values = col_val) + 
  scale_fill_manual(values = col_val) + 
  geom_rangeframe(data=data_frame, mapping=aes(x=x, y=y), color = "black", size = 1.0) + 
  labs(x = "", y = "") + 
  my_theme + theme(legend.position = "none")
ggsave(str_c(dir_save, "Param_pred.svg"), p, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)


# Get some aggregated data 
data_c1_sum <- data_param %>% filter(c1 < 15)
summary(c1_val)
ggplot(data_c1_sum, aes(c1)) + 
  geom_histogram(bins = 50)
