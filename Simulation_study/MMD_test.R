

# install.packages("Ecume")
# install.packages("Peacock.test")

library(Ecume)
library(rstudioapi)
library(Peacock.test)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

ind = 4

sample_origin = as.matrix(read.csv(paste0("sample_", ind, ".csv")))
gAE_recon = as.matrix(read.csv(paste0("gAE_recon_", ind, ".csv")))
VAE_recon = as.matrix(read.csv(paste0("VAE_recon_", ind, ".csv")))
gAE_gen = as.matrix(read.csv(paste0("gAE_gen_", ind, ".csv")))
VAE_gen = as.matrix(read.csv(paste0("VAE_gen_", ind, ".csv")))

mmd_test(sample_origin, gAE_recon, type = "unbiased")
mmd_test(sample_origin, VAE_recon, type = "unbiased")
mmd_test(sample_origin, gAE_gen, type = "unbiased")
mmd_test(sample_origin, VAE_gen, type = "unbiased")

mmd_test(sample_origin, gAE_recon, type = "linear")
mmd_test(sample_origin, VAE_recon, type = "linear")
mmd_test(sample_origin, gAE_gen, type = "linear")
mmd_test(sample_origin, VAE_gen, type = "linear")

# peacock2 : two-dimensional Kolmogorov-Smirnov test
peacock2(sample_origin, gAE_recon)
peacock2(sample_origin, VAE_recon)
peacock2(sample_origin, gAE_gen)
peacock2(sample_origin, VAE_gen)
