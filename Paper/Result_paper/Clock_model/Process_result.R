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

dir_save <- "../../../Results/Paper_figures/Fig2/"
if(!dir.exists(dir_save)) dir.create(dir_save)

dir_res <- "../../../Intermediate/Multiple_individuals/Clock_model/Ram_sampler/Npart1000_nsamp50000_corr0.0_exp_id1_run1/"

data_mean <- read_csv(str_c(dir_res, "Mean.csv"), col_types = cols()) %>%
  mutate(sample = 1:50000) %>%
  filter(sample > 10000 * 0.2) #%>%
  filter(mu1 > 0.5) %>%
  filter(mu2 > 2.5) %>%
  filter(mu3 < 2.0)

p1 <- ggplot(data_mean, aes(mu1)) + 
  geom_density(size = 3.0, color = my_colors[8]) + 
  geom_rangeframe(size = 3.0, color = "black", linetype = 1) + 
  geom_vline(xintercept = log(3.0), size = 2.0) + 
  labs(x = "", y = "") + 
  my_theme + theme(legend.position = "none", text = element_text(size = 40))
p2 <- ggplot(data_mean, aes(mu2)) + 
  geom_density(size = 3.0, color = my_colors[8]) + 
  geom_rangeframe(size = 3.0, color = "black", linetype = 1) + 
  geom_vline(xintercept = log(30.0), size = 2.0) + 
  labs(x = "", y = "") + 
  my_theme + theme(legend.position = "none", text = element_text(size = 40))
p3 <- ggplot(data_mean, aes(mu3)) + 
  geom_density(size = 3.0, color = my_colors[8]) + 
  geom_rangeframe(size = 3.0, color = "black", linetype = 1) + 
  geom_vline(xintercept = log(3.0), size = 2.0) + 
  labs(x = "", y = "") + 
  my_theme + theme(legend.position = "none", text = element_text(size = 40))
p4 <- ggplot(data_mean, aes(mu4)) + 
  geom_density(size = 3.0, color = my_colors[8]) + 
  geom_rangeframe(size = 3.0, color = "black", linetype = 1) + 
  geom_vline(xintercept = log(2.0), size = 2.0) + 
  labs(x = "", y = "") + 
  my_theme + theme(legend.position = "none", text = element_text(size = 40))

ggsave(str_c(dir_save, "Mu1.svg"), p1, bg = "transparent", width = BASE_WIDTH-1, height = BASE_HEIGHT)
ggsave(str_c(dir_save, "Mu2.svg"), p2, bg = "transparent", width = BASE_WIDTH-1, height = BASE_HEIGHT)
ggsave(str_c(dir_save, "Mu3.svg"), p3, bg = "transparent", width = BASE_WIDTH-1, height = BASE_HEIGHT)
ggsave(str_c(dir_save, "Mu4.svg"), p4, bg = "transparent", width = BASE_WIDTH-1, height = BASE_HEIGHT)

data_scale <- read_csv(str_c(dir_res, "Scale.csv"), col_types = cols()) %>%
  mutate(sample = 1:50000) #%>%
  filter(sample > 10000 * 0.2) %>%
  filter(scale1 < 1.0) %>%
  filter(scale2 < 1) %>%
  filter(scale3 < 1)

p1 <- ggplot(data_scale, aes(scale1)) + 
  geom_density(size = 3.0, color = my_colors[8]) + 
  geom_rangeframe(size = 3.0, color = "black", linetype = 1) + 
  geom_vline(xintercept = 0.4, size = 2.0) + 
  labs(x = "", y = "") + 
  my_theme + theme(legend.position = "none", text = element_text(size = 40))
p2 <- ggplot(data_scale, aes(scale2)) + 
  geom_density(size = 3.0, color = my_colors[8]) + 
  geom_rangeframe(size = 3.0, color = "black", linetype = 1) + 
  geom_vline(xintercept = 0.2, size = 2.0) + 
  labs(x = "", y = "") + 
  my_theme + theme(legend.position = "none", text = element_text(size = 40))
p3 <- ggplot(data_scale, aes(scale3)) + 
  geom_density(size = 3.0, color = my_colors[8]) + 
  geom_rangeframe(size = 3.0, color = "black", linetype = 1) + 
  geom_vline(xintercept = 0.2, size = 2.0) + 
  labs(x = "", y = "") + 
  my_theme + theme(legend.position = "none", text = element_text(size = 40))
p4 <- ggplot(data_scale, aes(scale4)) + 
  geom_density(size = 3.0, color = my_colors[8]) + 
  geom_rangeframe(size = 3.0, color = "black", linetype = 1) + 
  geom_vline(xintercept = 0.1, size = 2.0) + 
  labs(x = "", y = "") + 
  my_theme + theme(legend.position = "none", text = element_text(size = 40))

ggsave(str_c(dir_save, "Scale1.svg"), p1, bg = "transparent", width = BASE_WIDTH-1, height = BASE_HEIGHT)
ggsave(str_c(dir_save, "Scale2.svg"), p2, bg = "transparent", width = BASE_WIDTH-1, height = BASE_HEIGHT)
ggsave(str_c(dir_save, "Scale3.svg"), p3, bg = "transparent", width = BASE_WIDTH-1, height = BASE_HEIGHT)
ggsave(str_c(dir_save, "Scale4.svg"), p4, bg = "transparent", width = BASE_WIDTH-1, height = BASE_HEIGHT)

  path_data <-  "../../../Intermediate/Simulated_data/SSA/Multiple_ind/Clock/Clock.csv"
data_obs <- read_csv(path_data, col_types = cols(id = col_factor()))
data_rand <- data_obs %>%
  filter(id %in% c(1, 19, 2, 32))
data_obs_quant <- data_obs %>%
  group_by(time) %>%
  summarise(median = median(obs), 
            quant05 = quantile(obs, 0.05), 
            quant95 = quantile(obs, 0.95))
frame_data <- tibble(x = c(0, 50), y = c(0, 150))

p <- ggplot(data_obs_quant, aes(time, median)) + 
  geom_line(data=data_obs, mapping = aes(time, obs, group=id), color = my_colors[1], size = 0.1) +
  geom_line(data=data_rand, mapping = aes(time, obs, group = id), color = "#444444", size=0.9) + 
  geom_rangeframe(data=frame_data, mapping=aes(x = x, y = y), size = 1.0) + 
  labs(x = "", y = "") + 
  my_theme
ggsave(str_c(dir_save, "Obs_data.svg"), p, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)

# Plotting the pvc
data_quant <- read_csv(str_c(dir_res, "Pvc.csv"), col_types = cols())

lw <- 1.0
p <- ggplot(data_obs_quant, aes(time, median)) + 
  geom_line(data=data_obs, mapping = aes(time, obs, group=id), color = my_colors[1], size = 0.1) +
  #geom_line(data=data_rand, mapping = aes(time, exp(obs), group = id), color = "#666666", size=0.9) + 
  geom_ribbon(aes(ymin = quant05, ymax = quant95), alpha=0.0, color = "black", size = lw) + 
  geom_line(data=data_quant, aes(time, y1_med), color = my_colors[4], size = lw) + 
  geom_line(data=data_quant, aes(time, y1_qu05), color = my_colors[4], size = lw) + 
  geom_line(data=data_quant, aes(time, y1_qu95), color = my_colors[4], size = lw) + 
  geom_line(size = 1.5, linetype = 1) + 
  geom_rangeframe(data=frame_data, mapping=aes(x = x, y = y), size = 1.0) +
  labs(x = "", y = "") + 
  my_theme
ggsave(str_c(dir_save, "Data_pvc.svg"), p, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)


data_pvc_quant <- read_csv(str_c(dir_res, "Pvc_quant.csv"), col_types = cols())
p <- ggplot(data_pvc_quant, aes(x = time)) + 
  geom_ribbon(aes(x=time, ymin = y1_med_low, ymax = y1_med_upp), alpha = 0.15) + 
  geom_ribbon(aes(x=time, ymin = y1_qu05_low, ymax = y1_qu05_upp), alpha = 0.15) + 
  geom_ribbon(aes(x=time, ymin = y1_qu95_low, ymax = y1_qu95_upp), alpha = 0.15) + 
  geom_line(data=data_obs_quant, mapping=aes(time, median), size = lw, color = "#444444") + 
  geom_line(data=data_obs_quant, mapping=aes(time, quant05), color = "#444444", size = lw) + 
  geom_line(data=data_obs_quant, mapping=aes(time, quant95), color = "#444444", size = lw) + 
  geom_rangeframe(data=frame_data, mapping=aes(x = x, y = y), size = 1.0) +
  labs(x = "", y = "") + 
  my_theme
ggsave(str_c(dir_save, "Data_pvc_quant.svg"), p, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)

# Create prediction plot 
data_sim_path <- "../../Intermediate//Multiple_individuals/Clock_alt/Ram_sampler/Pvc_quant.csv"
data_sim <- read_csv(data_sim_path, col_types = cols())  

data_path100 <- "../../Intermediate/Simulated_data/SSA/Multiple_ind/Clock_predict/Ex2_100000.csv"
data_100 <- read_csv(data_path100, col_types = cols()) 
n_time_points <- 64
labels <- rep(1:2500, each = n_time_points * 40)
data_100 <- data_100 %>%
  mutate(label = labels) %>%
  group_by(time, label) %>%
  summarise(median = median(obs), 
            quant975 = quantile(obs, 0.975), 
            quant025 = quantile(obs, 0.025))

data_100_sum <- data_100 %>%
  group_by(time) %>%
  summarise(median_low = quantile(median, 0.025), 
            median_upp = quantile(median, 0.975), 
            quant975_low = quantile(quant975, 0.025), 
            quant975_upp = quantile(quant975, 0.975), 
            quant025_low = quantile(quant025, 0.025), 
            quant025_upp = quantile(quant025, 0.975), 
            median = median(median), 
            quant025 = median(quant025), 
            quant975 = median(quant975))
index <- seq(from = 1, by = 3, to = 64)
data_100_sum <- data_100_sum[index, ]

range_data <- tibble(x = c(0, 50), y = c(0, 250))
p <- ggplot(data_sim) + 
  geom_ribbon(aes(x = time, ymin = y1_med_low, ymax = y1_med_upp), alpha = 0.15) + 
  geom_ribbon(aes(x = time, ymin = y1_qu05_low, ymax = y1_qu05_upp), alpha = 0.15) + 
  geom_ribbon(aes(x = time, ymin = y1_qu95_low, ymax = y1_qu95_upp), alpha = 0.15) + 
  geom_linerange(data=data_100_sum, mapping=aes(time, ymin = median_low, ymax=median_upp), color = "#444444") +
  geom_linerange(data=data_100_sum, mapping=aes(time, ymin = quant975_low, ymax=quant975_upp), color = "#444444") + 
  geom_linerange(data=data_100_sum, mapping=aes(time, ymin = quant025_low, ymax=quant025_upp), color = "#444444") + 
  geom_line(data=data_100_sum, mapping=aes(time, median), color = "#444444", size = lw) + 
  geom_line(data=data_100_sum, mapping=aes(time, quant025), color = "#444444", size = lw) + 
  geom_line(data=data_100_sum, mapping=aes(time, quant975), color = "#444444", size = lw) + 
  geom_rangeframe(data=range_data, mapping=aes(x, y), size = 1.0) + 
  labs(y = "", x = "") + 
  my_theme
ggsave(str_c(dir_save, "Data_predict.svg"), p, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)


# Plot for the supplmentary 
data_corr <- read_csv(str_c(dir_res, "Corr.csv"), col_types = cols())
n_corr_param <- 4
n_samples <- 50000
corr_tibble <- tibble(samples = 1:50000)
for(i in 1:n_corr_param-1){
  
  if(i <= 0) next 
  
  for(j in (i+1):n_corr_param){
    index_choose <- seq(from = i, by = n_corr_param, length.out = n_samples)
    corr_tibble[str_c("c", as.character(i), as.character(j))] <- data_corr[index_choose, j]
  }
}


p1 <- ggplot(corr_tibble, aes(c12)) + 
  geom_density(linetype = 1, size = 2.0, color = "#EBD5A3") + 
  geom_vline(xintercept = -0.16, size = 1.0) + 
  labs(x = "", y = "") + 
  geom_rangeframe(size = 1.0) + 
  my_theme
p2 <- ggplot(corr_tibble, aes(c13)) + 
  geom_density(linetype = 1, size = 2.0, color = "#EBD5A3") + 
  geom_vline(xintercept = -0.17, size = 1.0) + 
  labs(x = "", y = "") + 
  geom_rangeframe(size = 1.0) + 
  my_theme
p3 <- ggplot(corr_tibble, aes(c14)) + 
  geom_density(linetype = 1, size = 2.0, color = "#EBD5A3") + 
  geom_vline(xintercept = -0.14, size = 1.0) + 
  labs(x = "", y = "") + 
  geom_rangeframe(size = 1.0) + 
  my_theme
p4 <- ggplot(corr_tibble, aes(c23)) + 
  geom_density(linetype = 1, size = 2.0, color = "#EBD5A3") + 
  geom_vline(xintercept = -0.26, size = 1.0) + 
  labs(x = "", y = "") + 
  geom_rangeframe(size = 1.0) + 
  my_theme
p5 <- ggplot(corr_tibble, aes(c24)) + 
  geom_density(linetype = 1, size = 2.0, color = "#EBD5A3") + 
  geom_vline(xintercept = -0.22, size = 1.0) + 
  labs(x = "", y = "") + 
  geom_rangeframe(size = 1.0) + 
  my_theme
p6 <- ggplot(corr_tibble, aes(c34)) + 
  geom_density(linetype = 1, size = 2.0, color = "#EBD5A3") + 
  geom_vline(xintercept = -0.49, size = 1.0) + 
  labs(x = "", y = "") + 
  geom_rangeframe(size = 1.0) + 
  my_theme

ggsave(str_c(dir_save, "Corr12.svg"), p1, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)
ggsave(str_c(dir_save, "Corr13.svg"), p2, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)
ggsave(str_c(dir_save, "Corr14.svg"), p3, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)
ggsave(str_c(dir_save, "Corr23.svg"), p4, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)
ggsave(str_c(dir_save, "Corr24.svg"), p5, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)
ggsave(str_c(dir_save, "Corr34.svg"), p6, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)
