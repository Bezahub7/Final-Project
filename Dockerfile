# start from the rstudio/plumber image
FROM rstudio/plumber

# install the linux libraries needed for plumber
RUN apt-get update -qq && apt-get install -y  libssl-dev  libcurl4-gnutls-dev  libpng-dev pandoc 

#directory
WORKDIR /app
    
# install packages
RUN R -e "install.packages(c('tidymodels', 'readr', 'dplyr','plumber', 'yardstick'))"

# copy myAPI.R from the current directory into the container
COPY API.R /app/API.R

# Copy my dataset 
COPY data/diabetes_binary_health_indicators_BRFSS2015.csv /app/data/diabetes_binary_health_indicators_BRFSS2015.csv


# open port to traffic
EXPOSE 8000

# when the container starts, start the myAPI.R script
ENTRYPOINT ["R", "-e", \
    "pr <- plumber::plumb('API.R'); pr$run(host='0.0.0.0', port=8000)"]
