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

dir_save <- "../../../Results/Paper_figures/FigS1/"
if(!dir.exists(dir_save)) dir.create(dir_save)

dir_data1 <- "../../../Intermediate/Multiple_individuals/Ornstein_val_opt1/Gen_am_sampler/Npart1000_nsamp60000_corr0.99_exp_id1_run1/"
dir_data_alt <- "../../../Intermediate/Multiple_individuals/Ornstein_val_opt2/Gen_am_sampler/Npart1000_nsamp60000_corr0.99_exp_id1_run1/"
dir_val <- "../../../Intermediate/ou_data/"

data_val <- read_csv(str_c(dir_val, "chain_eta.csv"), col_types = cols()) 
colnames(data_val) <- c("mu1", "mu2", "mu3", "scale1", "scale2", "scale3")
data_val <- data_val %>%
  mutate(sample = 1:60000, data_set = "val")


mean1 <- read_csv(str_c(dir_data1, "Mean.csv"), col_types = cols()) %>%
  mutate(sample = 1:60000, data_set = "stand")
mean2 <- read_csv(str_c(dir_data_alt, "Mean.csv"), col_types = cols()) %>%
  mutate(sample = 1:60000, data_set = "alt")
mean3 <- data_val %>%
  select(mu1, mu2, mu3, sample, data_set)


data_mean <- mean1 %>%
  bind_rows(mean2) %>% 
  bind_rows(mean3) %>%
  mutate(data_set = as.factor(data_set))

range_data1 <- tibble(x = c(-1.2, -0.4), y = c(0, 6))
p1 <- ggplot(data_mean, aes(mu1)) + 
  geom_density(aes(color = data_set, linetype = data_set), size = 2.0) + 
  scale_color_manual(values = my_colors[-1]) + 
  xlim(-1.2, -0.4) + 
  geom_rangeframe(data=range_data1, mapping=aes(x, y), color = "black", size = 2.0) + 
  geom_vline(xintercept = -0.7, size = 2.0) + 
  labs(x = "", y = "") + 
  my_theme
range_data2 <- tibble(x = c(2.0, 2.6), y = c(0, 6))
p2 <- ggplot(data_mean, aes(mu2)) + 
  geom_density(aes(color = data_set, linetype = data_set), size = 2.0) + 
  geom_rangeframe(data=range_data2, mapping=aes(x, y), color = "black", size = 2.0) + 
  xlim(2.0, 2.6) + 
  labs(x = "", y = "") + 
  scale_color_manual(values = my_colors[-1]) + 
  geom_vline(xintercept = 2.3, size = 2.0) + 
  my_theme
range_data3 <- tibble(x = c(-1.4, -0.4), y = c(0, 6))
p3 <- ggplot(data_mean, aes(mu3)) + 
  geom_density(aes(color = data_set, linetype = data_set), size = 2.0) + 
  geom_rangeframe(data=range_data3, mapping=aes(x, y), color = "black", size = 2.0) + 
  geom_vline(xintercept = -0.9, size = 2.0) + 
  xlim(-1.4, -0.4) + 
  labs(x = "", y = "") + 
  scale_color_manual(values = my_colors[-1]) + 
  my_theme

p_mean <- ggpubr::ggarrange(p1, p2, p3, ncol = 3, common.legend = T, legend = "bottom")
ggsave(str_c(dir_save, "Mean.svg"), p_mean, bg = "transparent", width = BASE_WIDTH*3, height = BASE_HEIGHT)


scale1 <- read_csv(str_c(dir_data1, "Scale.csv"), col_types = cols()) %>%
  mutate(sample = 1:60000, data_set = "stand")
scale2 <- read_csv(str_c(dir_data_alt, "Scale.csv"), col_types = cols()) %>%
  mutate(sample = 1:60000, data_set = "alt")
scale3 <- data_val %>%
  select(scale1, scale2, scale3, sample, data_set)

data_scale <- scale1 %>%
  bind_rows(scale2) %>% 
  bind_rows(scale3) %>%
  mutate(data_set = as.factor(data_set)) %>%
  filter(sample > 12000)

range_data1 <- tibble(x = c(0, 10), y = c(0, 0.5))
p1 <- ggplot(data_scale, aes(scale1)) + 
  geom_density(aes(color = data_set, linetype = data_set), size = 2.0) + 
  geom_rangeframe(data=range_data1, mapping=aes(x, y), color = "black", size = 2.0) + 
  geom_vline(xintercept = 4.0, size = 2.0) + 
  xlim(0, 10) + 
  labs(x = "", y = "") + 
  scale_color_manual(values = my_colors[-1]) + 
  my_theme
range_data1 <- tibble(x = c(0, 12), y = c(0, 0.5))
p2 <- ggplot(data_scale, aes(scale2)) + 
  geom_density(aes(color = data_set, linetype = data_set), size = 2.0) + 
  geom_rangeframe(data=range_data1, mapping=aes(x, y), color = "black", size = 2.0) + 
  geom_vline(xintercept = 10.0, size = 2.0) + 
  xlim(0, 12) + 
  labs(x = "", y = "") + 
  scale_color_manual(values = my_colors[-1]) + 
  my_theme
range_data1 <- tibble(x = c(0, 10), y = c(0, 0.5))
p3 <- ggplot(data_scale, aes(scale3)) + 
  geom_density(aes(color = data_set, linetype = data_set), size = 2.0) + 
  geom_rangeframe(data=range_data1, mapping=aes(x, y), color = "black", size = 2.0) + 
  geom_vline(xintercept = 4.0, size = 2.0) + 
  xlim(0, 10) + 
  labs(x = "", y = "") + 
  scale_color_manual(values = my_colors[-1]) + 
  my_theme

p_scale <- ggpubr::ggarrange(p1, p2, p3, ncol = 3, common.legend = T, legend = "bottom")
ggsave(str_c(dir_save, "Scale.svg"), p_scale, bg = "transparent", width = BASE_WIDTH*3, height = BASE_HEIGHT)

sigma1 <- read_csv(str_c(dir_data1, "Kappa_sigma.csv"), col_types = cols()) %>%
  mutate(sample = 1:60000, data_set = "stand")
sigma2 <- read_csv(str_c(dir_data_alt, "Kappa_sigma.csv"), col_types = cols()) %>%
  mutate(sample = 1:60000, data_set = "alt")
sigma3 <- read_csv(str_c(dir_val, "chain_sigma_epsilon.csv"), col_types = cols())  %>%
  mutate(sample = 1:60000, data_set = "val") %>%
  rename("sigma1" = "x1")

data_sigma <- sigma1 %>%
  bind_rows(sigma2) %>%
  bind_rows(sigma3) %>%
  mutate(data_set = as.factor(data_set)) %>%
  filter(sample > 10000)

range_data1 <- tibble(x = c(0.28, 0.32), y = c(0, 150))
p_sigma <- ggplot(data_sigma, aes(sigma1)) + 
  geom_density(aes(color = data_set, linetype = data_set), size = 2.0) + 
  geom_rangeframe(data=range_data1, mapping=aes(x, y), color = "black", size = 2.0) + 
  geom_vline(xintercept = 0.3, size = 2.0) + 
  xlim(0.28, 0.32) + 
  labs(x = "", y = "") + 
  scale_color_manual(values = my_colors[-1]) + 
  my_theme + theme(legend.position = "none")

ggsave(str_c(dir_save, "Sigma.svg"), p_sigma, width = BASE_WIDTH, height = BASE_HEIGHT)
