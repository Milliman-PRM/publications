#' ## Code Owners: Ben Copeland
#' ### OWNERS ATTEST TO THE FOLLOWING:
#'   * The `master` branch will meet Milliman QRM standards at all times.
#'   * Deliveries will only be made from code in the `master` branch.
#'   * Review/Collaboration notes will be captured in Pull Requests (prior to merging).
#' 
#' 
#' ### Objective:
#'   * Make some pretty graphs for recommender model evaluation
#' 
#' ### Developer Notes:
#'   * <What future developers need to know.>

# load our packages
options(repos=structure(c(CRAN="http://cran.rstudio.com/")))
require(splines)
require(ggplot2)
require(scales)
require(RColorBrewer)
require(mgcv)
require(grid)
require(ifultools)
require(sparkR)

#' Call the R parser to include the meta parameters
source(paste(c(
  Sys.getenv('UserProfile')
  ,'/HealthBI_LocalData/Supp03_Parser.R'
),collapse=""),echo = TRUE)

print(dir.data <- args.modules["160","mod_out"])
setwd(dir.data)

# initialize sparkContext which starts a new Spark session
sc <- sparkR.init()

# initialize sqlContext
sq <- sparkRSQL.init(sc)

# load parquet file into a Spark data frame and coerce into R data frame
df <- collect(read.parquet(sq, paste0(dir.data, 'model_evaluation')))

plot.skel.chronic <- ggplot(
    df[df$chronic_flag == '1',]
    )

plot.chronic <- plot.skel.chronic + geom_point(
    alpha=1
    ,aes(x=pred_rank,y=cumul_pred, color=pred_type)
  ) + theme_bw() + scale_color_brewer(palette="Set1") + geom_line(
    aes(x=pred_rank,y=cumul_pred, color=pred_type)
  ) + scale_x_continuous(name='\nPrediction Rank', labels=seq(15), breaks=seq(15)) + 
  scale_y_continuous(name='Cumulative Percent\nPredicted\n', labels=seq(0,1,0.1),breaks=seq(0,1,0.1)) +
  labs(color='Model Type') + guides(guides='None') + theme(
      panel.border = element_blank()
      ,panel.grid.major = element_blank()
      ,panel.grid.minor = element_blank()
      ,axis.line = element_line(colour = "black")
  ) + ggtitle('Prediction Accuracy,\nChronic Conditions')
plot.chronic

plot.skel.non.chronic <- ggplot(
  df[df$chronic_flag == '0',]
)

plot.non.chronic <- plot.skel.non.chronic + geom_point(
  alpha=1
  ,aes(x=pred_rank,y=cumul_pred, color=pred_type)
) + theme_bw() + scale_color_brewer(palette="Set1") + geom_line(
  aes(x=pred_rank,y=cumul_pred, color=pred_type)
) + scale_x_continuous(name='\nPrediction Rank', labels=seq(15), breaks=seq(15)) + 
  scale_y_continuous(name='Cumulative Percent\nPredicted\n', labels=seq(0,1,0.1),breaks=seq(0,1,0.1)) +
  labs(color='Model Type') + guides(guides='None') + theme(
    panel.border = element_blank()
    ,panel.grid.major = element_blank()
    ,panel.grid.minor = element_blank()
    ,axis.line = element_line(colour = "black")
  ) + ggtitle('Prediction Accuracy,\nNon-Chronic Conditions')
plot.non.chronic

ggsave(
  paste0('eval_chronic.png')
  ,plot.chronic
  ,type='cairo-png'
  ,scale=.8
)

ggsave(
  paste0('eval_non_chronic.png')
  ,plot.non.chronic
  ,type='cairo-png'
  ,scale=.8
)


# terminate Spark session
sparkR.stop()
