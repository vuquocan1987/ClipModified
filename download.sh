# clip todownload list of files using wget

# file list
files = """

https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar
https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar
https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar
https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar
https://msd-for-monai.s3-us-west-2.amazonaws.com/Task05_Prostate.tar
https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar
https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar
https://msd-for-monai.s3-us-west-2.amazonaws.com/Task08_HepaticVessel.tar
https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar
https://msd-for-monai.s3-us-west-2.amazonaws.com/Task10_Colon.tar

# download files
wget -i todownload