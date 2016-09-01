#' ## Code Owners: Ben Copeland, Shea Parkes
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
require(dplyr)
require(magrittr)
require(ggplot2)
require(scales)
require(RColorBrewer)
require(mgcv)
require(grid)
require(ifultools)
require(SparkR)

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

plot.skel <- df %>%
  mutate(
    chronic_pretty = factor(
      chronic_flag
      ,levels=c(1, 0)
      ,labels=c('Chronic', 'Non-Chronic')
    )
    ,pred_type_pretty = factor(
      pred_type
      ,levels=c('recommender', 'popular')
      ,labels=c('Matrix\nFactorizaiton', 'Popularity')
    )
  ) %>%
  ggplot()

plot.facet <- plot.skel + geom_point(
    alpha=1
    ,aes(x=pred_rank,y=cumul_pred, color=pred_type_pretty)
  ) +
  facet_wrap(~chronic_pretty) +
  theme_bw() +
  scale_color_brewer(palette="Set1") +
  geom_line(
    aes(x=pred_rank,y=cumul_pred, color=pred_type_pretty)
  ) +
  scale_x_continuous(name='\nNumber of top-rated\nconditions per patient', breaks=c(1,5,10,15)) +
  scale_y_continuous(name='Percent of new\nconditions captured', labels=percent,breaks=seq(0,1,0.1), limits=c(0,0.7)) +
  labs(color='Model Type') +
  theme(
      #panel.border = element_blank()
      #panel.grid.major = element_blank()
      #panel.grid.minor = element_blank()
      axis.line = element_line(colour = "black")
  ) +
  ggtitle('Model accuracy\non two month hold-out')+
  coord_cartesian(ylim=c(0,0.73))
plot.facet


ggsave(
  paste0('eval_facet.png')
  ,plot.facet
  ,type='cairo-png'
  ,scale=1.1
  ,width=6
  ,height=4
)

ggsave(
  paste0('eval_facet_presentation.png')
  ,plot.facet
  ,type='cairo-png'
  ,scale=.75
  ,width=12
  ,height=8
)


# terminate Spark session
sparkR.stop()
