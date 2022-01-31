library(tidyverse)
library(ggthemes)
library(readxl)


# General plotting parameters 
my_theme <- theme_tufte(base_size = 24) + theme(plot.title = element_text(hjust = 0.5, size = 14, face="bold"), 
                                                plot.subtitle = element_text(hjust = 0.5)) +
  theme(axis.title=element_text(size=24)) + theme(rect = element_rect(fill = "transparent"))
my_colors <- c(
  "#999999", "#E69F00", "#56B4E9", "#009E73",
  "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
BASE_HEIGHT <- 5
BASE_WIDTH <- 7.0

dir_save <- "../../../Results/Paper_figures/Fig5/"
if(!dir.exists(dir_save)) dir.create(dir_save)


path_data1 <- "../../../Data/Fructose_data/WT_Fru_0to2.xlsx"
data1 <- read_excel(path_data1) 

path_data2 <- "../../../Data/Fructose_data/WT_Fru_0to0.05.xlsx"
data2 <- read_excel(path_data2) 

data1_plot <- data1 %>%
  select(Cell, ratio, TP, Cell_Area, Nuclear_Mean, Cell_Mean) %>%
  rename("id" = "Cell") %>%
  mutate(id = as.factor(id), TP = TP - 1) %>%
  mutate(ratio_new = (Nuclear_Mean - Cell_Mean) / (Cell_Mean)) %>%
  mutate(time = TP * 30) %>%
  mutate(time = time / 60) %>%
  mutate(fruc = 2.0) %>%
  filter(ratio_new > 0.01)

data2_plot <- data2 %>%
  select(Cell, ratio, TP, Cell_Area, Nuclear_Mean, Cell_Mean) %>%
  rename("id" = "Cell") %>%
  mutate(id = id + 22) %>%
  mutate(id = as.factor(id), TP = TP - 1) %>%
  mutate(ratio_new = (Nuclear_Mean - Cell_Mean) / (Cell_Mean)) %>%
  mutate(ratio_new = ratio_new + 0.06) %>% # Account BG-flourescence 
  mutate(time = TP * 30) %>%
  mutate(time = time / 60) %>%
  mutate(fruc = 0.05) 


# General plotting 0 -> 2.0
ggplot(data1_plot, aes(time, ratio_new, color = id)) + 
  geom_line(aes(group = id)) + 
  geom_point() + 
  geom_rangeframe(color = "black") + 
  labs(x = "Time [min]", y = "Ratio") +
  geom_vline(xintercept = 90 / 60) + 
  my_theme + theme(legend.position = "none")


# General plotting: 0 -> 0.05
ggplot(data2_plot, aes(time, ratio_new, color = id)) + 
  geom_line(aes(group = id)) + 
  geom_point() + 
  geom_rangeframe(color = "black") + 
  labs(x = "Time", y = "Ratio") +
  geom_vline(xintercept = 90 / 60) +
  my_theme + theme(legend.position = "none")

# Process data into full training data-set 
data_full <- data1_plot %>%
  bind_rows(data2_plot) %>%
  mutate(fruc = as.factor(fruc), id = as.factor(id))

ggplot(data_full, aes(time, ratio_new, color = fruc)) +
  geom_line(aes(group = id)) + 
  scale_color_brewer(palette = "Dark2") +
  geom_rangeframe(color = "black") + 
  my_theme

# Filter cells that behave wierd in microscope 
data_save <- data1_plot %>%
  bind_rows(data2_plot) %>%
  mutate(obs = exp(log(ratio_new))) %>%
  mutate(fruc = as.factor(fruc)) %>%
  mutate(frud_ic = fruc) %>%
  mutate(obs_id = 1) %>%
  filter(id != 24) %>%
  filter(id != 25) %>%
  filter(id != 41) %>%
  mutate(fruc_scale_fac = 4 / as.numeric(as.character(fruc))) %>%
  mutate(fruc_mM = 220 / fruc_scale_fac)

# Write to disk 
dir_save <- "../../../Intermediate/Experimental_data/Data_fructose/"
if(!dir.exists(dir_save)) dir.create(dir_save, recursive = T)
write_csv(data_save, str_c(dir_save, "Fructose_data.csv"))

# Save data with log-scale on rati
data_save_log <- data_save %>%
  mutate(obs = log(obs)) %>%
  filter(obs > -6) # Outlier observation 
write_csv(data_save_log, str_c(dir_save, "Fructose_data_log.csv"))


# Plot data for paper 
range_data <- tibble(x = c(0, 16), y = c(0, 0.55))
p1 <- ggplot(data1_plot, aes(time, ratio_new)) + 
  geom_line(aes(group=id), color = my_colors[1], size = 0.4) + 
  geom_rangeframe(data=range_data, mapping=aes(x = x, y = y), size = 1.0) +
  labs(x = "", y = "") + 
  my_theme

p2 <- ggplot(data2_plot, aes(time, ratio_new)) + 
  geom_line(aes(group=id), color = my_colors[1], size = 0.4) + 
  geom_rangeframe(data=range_data, mapping=aes(x = x, y = y), size = 1.0) +
  labs(x = "", y = "") + 
  my_theme

ggsave(str_c(dir_save, "Fru2.svg"), p1, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)
ggsave(str_c(dir_save, "Fru05.svg"), p2, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)
