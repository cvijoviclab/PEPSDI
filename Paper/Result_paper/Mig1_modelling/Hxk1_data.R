library(tidyverse)
library(ggthemes)
library(readxl)
library(data.table)


# General plotting parameters 
my_theme <- theme_tufte(base_size = 24) + theme(plot.title = element_text(hjust = 0.5, size = 14, face="bold"), 
                                                plot.subtitle = element_text(hjust = 0.5)) +
  theme(axis.title=element_text(size=24)) + theme(rect = element_rect(fill = "transparent"))
my_colors <- c(
  "#999999", "#E69F00", "#56B4E9", "#009E73",
  "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
BASE_HEIGHT <- 5
BASE_WIDTH <- 7.0


# Read Mig1-expression and Hxk1 data 
path_data <- "../../../Data/Fructose_data/20171130_CCv5.3_HexokinaseOverexpression_EtOHToFructose.xlsx"
data_hxk_raw <- read_excel(path_data, sheet = 2, na = "NaN")
data_ametrine <- read_excel(path_data, sheet = 1, na = "NaN")

# Process data into a tibble for plotting, starting with Hxk-expression 
n_col <- dim(data_hxk_raw)[2]
n_row <- dim(data_hxk_raw)[1]
strain_names <- unique(data_hxk_raw$`Strain / Time [min]`)
short_name <- c("delta_Hxk1_Hxk2", "delta_Hxk1_Hxk2+Hxk1", "delta_Hxk1_Hxk2+Hxk2", "delta_Hxk1_Hxk2+Glk1")

data_hxk_exp <- tibble()
time_vec <- as.numeric(colnames(data_ametrine[3:n_col]))
for(i in 1:n_row){
  
  strain_name <- short_name[which(data_ametrine$`Strain / Time [min]`[i] == strain_names)]
  
  data_tmp <- tibble(time = time_vec, 
                     response = as.numeric(data_ametrine[i, 3:n_col]), 
                     strain = strain_name, 
                     id = i)
  data_hxk_exp <- data_hxk_exp %>%
    bind_rows(data_tmp)
}


# Produce mean value between 240-480 min (short term hxk1-response)
data_hxk_exp <- data_hxk_exp %>%
  filter(time > 240) %>%
  filter(time < 480) %>%
  group_by(id) %>%
  summarise(mean_exp = median(response, na.rm = T))

# Join data with Mig1 data 
time_vec <- as.numeric(colnames(data_hxk_raw[3:n_col]))
data_plot <- tibble()
for(i in 1:n_row){

  strain_name <- short_name[which(data_hxk_raw$`Strain / Time [min]`[i] == strain_names)]
  
  data_tmp <- tibble(time = time_vec, 
                     response = as.numeric(data_hxk_raw[i, 3:n_col]), 
                     strain = strain_name, 
                     id = i, 
                     hxk_exp = data_hxk_exp$mean_exp[i])
  data_plot <- data_plot %>%
    bind_rows(data_tmp)
}

data_plot <- data_plot %>%
  mutate(id = as.factor(id), strain = as.factor(strain))

data_early_hxk1 <- data_plot %>%
  filter(time > 240 & time < 260) %>%
  mutate(time = as.factor(time)) %>%
  group_by(id, strain) %>%
  summarise(mean_mig1 = mean(response, na.rm = T), 
            mean_hxk  = mean(hxk_exp, na.rm = T)) %>%
  filter(strain == short_name[2]) 

  
p1 <- ggplot(data_early_hxk1, aes(mean_hxk, mean_mig1)) + 
  geom_point() + 
  geom_rangeframe() + 
  geom_smooth(method = "lm", se = F) + 
  my_theme 

# The linear relationship is strong 
my_mod <- lm(mean_mig1 ~ mean_hxk, data = data_early_hxk1)
(my_mod_sum <- summary(my_mod))

# qq-plot is good (not perfect but good)
qqnorm(my_mod$res)
qqline(my_mod$residuals)


# Access model simulations, take average over 15 minutes in model 
dir_best <- "../../../Intermediate/Multiple_individuals/Mig1_mod_2B/Ram_sampler/Npart100_nsamp40000_corr0.999_exp_id4_run1/"
data_plot <- read_csv(str_c(dir_best, "Pred_c1.csv"), col_types = cols()) %>%
  mutate(response = (t1 + t2 + t3) / 3) %>%
  filter(c1 < 10000)

# Get credabillity intervall for R2 predicted by the model 
R2_val <- tibble()
n_cell <- 132
for(i in 1:10000){

  i_min <- (i-1)*n_cell
  i_max <- (i)*n_cell
  #data_use <- data_plot[sample(1:(132*5000), n_cell),  ]
  data_use <- data_plot[i_min:i_max, ]  
  
  my_mod2 <- lm(response ~ c1, data = data_use)
  (sum_mod_2 <- summary(my_mod2))
  
  tmp <- tibble(R_val = sum_mod_2$r.squared)
  R2_val <- R2_val %>%
    bind_rows(tmp)
}

# Create the plots for the paper

# Linear relationship data 
range_data <- tibble(x = c(450, 2500), y = c(0.0, 1.25))
p1 <- ggplot(data_early_hxk1, aes(mean_hxk, mean_mig1)) + 
  geom_point(size = 3.0) + 
  geom_smooth(method = "lm", se = F) + 
  geom_rangeframe(data=range_data, mapping=aes(x, y), size = 1.0) + 
  labs(x = "", y = "") + 
  scale_fill_viridis_c() + 
  ylim(0, 1.25) + 
  my_theme 
p1

# Model simulations 
range_data <- tibble(x = c(0.0, 5000), y = c(0.0, 1.25))
p2 <- ggplot(data_plot, aes(c1, response)) + 
  geom_hex() + 
  scale_fill_viridis_c() + 
  geom_smooth(method = "lm", color = "blue", se = F) + 
  geom_rangeframe(data=range_data, mapping=aes(x, y), size = 1.0) + 
  ylim(0, 1.25) + 
  labs(x = "", y = "") + 
  my_theme + theme(legend.position = "none")
p2

# Predicted R2
p3 <- ggplot(R2_val, aes(R_val)) + 
  geom_histogram(bins = 50) + 
  geom_vline(xintercept = 0.094, size = 1.0) + 
  labs(x = "", y = "") +
  geom_rangeframe(size = 1.0) +
  my_theme

# For obtaining color bar when doing the figure 
p4 <- ggplot(data_plot, aes(c1, response)) + 
  geom_hex() + 
  scale_fill_viridis_c() + 
  geom_smooth(method = "lm", color = "blue", se = F) + 
  geom_rangeframe(data=range_data, mapping=aes(x, y), size = 1.0) + 
  ylim(0, 1.25) + 
  labs(x = "", y = "") + 
  my_theme
p4

dir_save_supp <- "../../../Results/Paper_figures/Fig6/Supp_fig/"
ggsave(str_c(dir_save_supp, "Hxk_val_data.svg"), p1, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)
ggsave(str_c(dir_save_supp, "Hxk_model_data.svg"), p2, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)
ggsave(str_c(dir_save_supp, "Histogram.svg"), p3, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)
ggsave(str_c(dir_save_supp, "Hxk_model_data_bar.svg"), p4, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)

