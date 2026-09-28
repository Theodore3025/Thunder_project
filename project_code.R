# NAME: Theodore Jun | NUMBER: 7637


# 1. Packages and data -------------------------------------------------------
# Install these packages once if needed; do not reinstall on every run.
# install.packages(c("readr", "dplyr", "ggplot2", "xgboost", "Matrix"))
library(readr)
library(dplyr)
library(ggplot2)
library(xgboost)
library(Matrix)

# Read data beside this script when using source() or Rscript.
# When running selected lines interactively, open the repository's R project first.
source_files <- vapply(sys.frames(), function(frame) {
  if (is.null(frame$ofile)) "" else as.character(frame$ofile)[1]
}, character(1))
script_argument <- grep("^--file=", commandArgs(), value = TRUE)
script_file <- if (any(nzchar(source_files))) {
  tail(source_files[nzchar(source_files)], 1)
} else if (length(script_argument) > 0) {
  sub("^--file=", "", script_argument[1])
} else {
  ""
}
data_path <- if (nzchar(script_file)) {
  dirname(normalizePath(script_file, mustWork = TRUE))
} else {
  getwd()
}

train <- read_csv(file.path(data_path, "training.csv.gz"), col_types = cols(.default = col_character()))
test <- read_csv(file.path(data_path, "testing.csv.gz"), col_types = cols(.default = col_character()))

# Training outcomes: 0 = miss, 1 = make.
train <- train %>% mutate(outcome = as.integer(as.logical(outcome)))
stopifnot(!anyNA(train$outcome), all(train$outcome %in% 0:1),
          !anyDuplicated(train$shot_id), !anyDuplicated(test$shot_id))

train <- train |>
  mutate(
    locationx = as.numeric(locationx),
    locationy = as.numeric(locationy),
    distance = as.numeric(distance),
    three = as.logical(three)
  )

test <- test |>
  mutate(
    locationx = as.numeric(locationx),
    locationy = as.numeric(locationy),
    distance = as.numeric(distance),
    three = as.logical(three)
  )


# 2. Helper functions --------------------------------------------------------
# Keep one copy of each helper; the final add_contest_features definition wins.

distance_to_three_line <- function(x, y) {
  arc_radius <- 23.75
  corner_y <- 22
  
  # Exact point where the straight corner line meets the curved arc.
  corner_end_x <- -41.75 + sqrt(arc_radius^2 - corner_y^2)
  
  # Express the shot's position relative to the basket.
  dx <- x + 41.75
  side_distance <- abs(y)
  
  # Find the nearest point on the curved portion.
  # Both sides of the court are symmetrical, so abs(y) works for either side.
  angle <- atan2(side_distance, dx)
  
  nearest_arc_angle <- pmin(
    angle,
    asin(corner_y / arc_radius)
  )
  
  arc_gap <- sqrt(
    (dx - arc_radius * cos(nearest_arc_angle))^2 +
      (side_distance - arc_radius * sin(nearest_arc_angle))^2
  )
  
  # Find the nearest point on the straight corner portion.
  nearest_corner_x <- pmax(-47, pmin(x, corner_end_x))
  
  corner_gap <- sqrt(
    (x - nearest_corner_x)^2 +
      (side_distance - corner_y)^2
  )
  
  # For each shot, return whichever distance is smaller.
  pmin(arc_gap, corner_gap)
}
add_shot_zones <- function(data) {
  
  data %>%
    mutate(
      
      # Distance from the basket at (-41.75, 0).
      rim_distance = sqrt(
        (locationx + 41.75)^2 + locationy^2
      ),
      
      # Shortest distance to the three-point line.
      three_line_gap = distance_to_three_line(
        locationx,
        locationy
      ),
      
      # Rules run from top to bottom.
      # Each shot receives the FIRST matching label.
      zone = case_when(
        
        # Missing information needed to classify the shot.
        is.na(locationx) | is.na(locationy) | is.na(three) ~
          "Unknown",
        
        # Coordinates outside the full court.
        # Half court is x = 0; the opposite baseline is x = 47.
        locationx < -47 |
          locationx > 47 |
          abs(locationy) > 25 ~
          "Outside court",
        
        # Shots taken on the half-court line or farther away.
        locationx >= 0 ~
          "Half court or beyond",
        
        # Three-point shots at least 3 feet beyond the line.
        # A tiny tolerance handles rounding at exactly 3 feet.
        three & three_line_gap >= 3 - 1e-9 ~
          "Deep three",
        
        # Two-point shots within a 5-foot radius of the basket.
        !three & rim_distance <= 5 ~
          "At the rim",
        
        # Remaining shots inside the paint.
        # Rim shots have already been assigned above.
        !three &
          locationx >= -47 &
          locationx < -28 &
          abs(locationy) <= 8 ~
          "Paint outside rim",
        
        # Two-point area from the free-throw line toward the arc.
        !three & locationx >= -28 ~
          "Free-throw-line area",
        
        # Two-point shots on either side of the paint.
        !three & locationx < -28 & locationy < -8 ~
          "Right midrange",
        
        !three & locationx < -28 & locationy > 8 ~
          "Left midrange",
        
        # Corner threes.
        three & locationx <= -32.8 & locationy < 0 ~
          "Right corner three",
        
        three & locationx <= -32.8 & locationy >= 0 ~
          "Left corner three",
        
        # Wing threes.
        three &
          locationx > -32.8 &
          locationx < -18 &
          locationy < 0 ~
          "Right wing three",
        
        three &
          locationx > -32.8 &
          locationx < -18 &
          locationy >= 0 ~
          "Left wing three",
        
        # Threes from the top of the key.
        three & locationx >= -18 ~
          "Top-of-the-key three",
        
        # Anything not covered by the rules above.
        TRUE ~ "Unknown"
      )
    )
}
log_loss <- function(actual, predicted) {
  
  # Avoid taking log(0) for predictions exactly equal to 0 or 1.
  p <- pmin(pmax(predicted, 1e-15), 1 - 1e-15)
  
  -mean(
    actual * log(p) +
      (1 - actual) * log1p(-p)
  )
}
add_shooter_features <- function(reference_data, target_data, k = 200) {
  
  # Overall make probability, used if a zone is unseen.
  global_prob <- mean(reference_data$outcome)
  
  # Zone averages calculated ONLY from reference_data.
  zone_lookup <- reference_data %>%
    group_by(zone) %>%
    summarise(
      zone_prob = mean(outcome),
      .groups = "drop"
    )
  
  # Player-zone counts calculated ONLY from reference_data.
  shooter_lookup <- reference_data %>%
    group_by(shooter_id, zone) %>%
    summarise(
      shooter_zone_attempts = n(),
      shooter_zone_makes = sum(outcome),
      .groups = "drop"
    )
  
  # Attach the reference statistics to the target shots.
  target_data %>%
    left_join(zone_lookup, by = "zone") %>%
    left_join(
      shooter_lookup,
      by = c("shooter_id", "zone")
    ) %>%
    mutate(
      
      # An unseen zone falls back to the overall average.
      zone_prob = coalesce(zone_prob, global_prob),
      
      # An unseen player-zone combination has zero known attempts/makes.
      shooter_zone_attempts = coalesce(shooter_zone_attempts, 0L),
      shooter_zone_makes = coalesce(shooter_zone_makes, 0L),
      
      # Shrink toward the zone average.
      # With zero attempts, this automatically equals zone_prob.
      shooter_zone_prob =
        (shooter_zone_makes + k * zone_prob) /
        (shooter_zone_attempts + k)
      
    ) %>%
    select(-shooter_zone_makes)
}
parse_approach <- function(value) {
  
  # Preserve missing entries as four missing measurements.
  if (is.na(value) || trimws(value) == "") {
    return(rep(NA_real_, 4))
  }
  
  # Remove the opening and closing braces.
  cleaned <- gsub("[{}]", "", value)
  
  # Split the text at each comma.
  pieces <- strsplit(cleaned, ",", fixed = TRUE)[[1]]
  
  # Each shot should contain exactly four measurements.
  if (length(pieces) != 4) {
    stop("An approach entry does not contain four distances.")
  }
  
  # Convert the four text values into numbers.
  as.numeric(pieces)
}
add_defender_approach <- function(data) {
  
  # Apply our helper to every shot.
  # numeric(4) means each entry must return four numbers.
  # t() arranges the result into one row per shot.
  approach_values <- t(
    vapply(
      data$closestdefapproach,
      parse_approach,
      numeric(4)
    )
  )
  
  data %>%
    mutate(
      
      # Ensure release distance is numeric in both datasets.
      closestdefdist = as.numeric(closestdefdist),
      
      # Extract each column of measurements.
      def_dist_1s = approach_values[, 1],
      def_dist_075s = approach_values[, 2],
      def_dist_05s = approach_values[, 3],
      def_dist_025s = approach_values[, 4],
      
      # Average gap-closing speed over the final second.
      closing_speed_1s =
        (def_dist_1s - closestdefdist) / 1,
      
      # Average gap-closing speed over the final quarter-second.
      closing_speed_last_025s =
        (def_dist_025s - closestdefdist) / 0.25
    )
}
add_shooter_movement <- function(data) {
  
  data %>%
    mutate(
      # Shooter's speed at release, in feet per second.
      shooterspeed = as.numeric(shooterspeed),
      
      # Number of dribbles before the shot.
      dribblesbefore = as.numeric(dribblesbefore),
      
      # TRUE if the shooter took no dribbles; FALSE otherwise.
      # Missing dribble counts remain missing.
      no_dribbles = dribblesbefore == 0
    )
}
make_oof_features <- function(data) {
  
  set.seed(2027)
  
  fold_data <- data %>%
    ungroup() %>%
    mutate(
      row_id = row_number(),
      fold_id = sample(rep(1:5, length.out = n()))
    )
  
  results <- vector("list", 5)
  
  for (fold in 1:5) {
    
    reference_rows <- fold_data %>%
      filter(fold_id != fold)
    
    target_rows <- fold_data %>%
      filter(fold_id == fold)
    
    results[[fold]] <- add_shooter_features(
      reference_data = reference_rows,
      target_data = target_rows,
      k = 200
    )
  }
  
  bind_rows(results) %>%
    arrange(row_id)
}
prepare_contest_columns <- function(data) {
  
  data %>%
    mutate(
      contested = as.logical(contested),
      
      num_contesters = as.integer(num_contesters),
      
      # Apply the same numeric conversion to all four distances.
      across(
        c(distcont1, distcont2, distcont3, distcont4),
        as.numeric
      )
    )
}
add_contest_features <- function(data) {
  
  data %>%
    mutate(
      
      # Group the sparse three- and four-contester cases together.
      contest_group = case_when(
        num_contesters == 0 ~ "None",
        num_contesters == 1 ~ "One",
        num_contesters == 2 ~ "Two",
        num_contesters >= 3 ~ "Three or more",
        TRUE ~ NA_character_
      ),
      
      contest_group = factor(
        contest_group,
        levels = c("None", "One", "Two", "Three or more")
      ),
      
      # Smallest available distance for each shot.
      nearest_contest_dist = pmin(
        distcont1,
        distcont2,
        distcont3,
        distcont4,
        na.rm = TRUE
      ),
      
      # log1p(x) means log(1 + x).
      # For uncontested shots, switch this distance term off with zero.
      contest_log_distance = if_else(
        num_contesters == 0,
        0,
        log1p(nearest_contest_dist)
      )
    )
}
prepare_reverse_context <- function(data) {
  
  data %>%
    prepare_contest_columns() %>%
    add_contest_features() %>%
    mutate(
      shotclock = as.numeric(shotclock),
      
      gamestate = factor(
        gamestate,
        levels = game_state_levels
      )
    )
}
add_precise_location <- function(data) {
  data %>%
    mutate(
      locationx = as.numeric(locationx),
      locationy = as.numeric(locationy),
      
      # Distance sideways from the court's center line.
      # For example, y = -8 and y = 8 both give 8 feet.
      lateral_distance = abs(locationy)
    )
}

# 3. Shot zones and descriptive summaries ------------------------------------

train <- add_shot_zones(train)
test <- add_shot_zones(test)
# Count the shots assigned to each zone.
train %>%
  count(zone, sort = TRUE)

test %>%
  count(zone, sort = TRUE)

# Inspect any shots that weren't classified.
train %>%
  filter(zone == "Unknown") %>%
  select(shot_id, locationx, locationy, three)

# Descriptive statistics only; these full-data averages are NOT model features.
zone_percentages <- train %>%
  group_by(zone) %>%
  summarise(
    attempts = n(),
    makes = sum(outcome),
    misses = attempts - makes,
    make_pct = 100 * mean(outcome),
    .groups = "drop"
  ) %>%
  arrange(desc(make_pct))
print(zone_percentages)

# Retain the original shot-map code for inspecting the custom zones.
# Makes the random sample repeatable.
set.seed(2027)

plot_data <- train %>%
  filter(
    zone != "Unknown",
    zone != "Outside court"
  ) %>%
  slice_sample(n = 5000)

# Angles covered by the curved portion of the three-point line.
arc_angles <- seq(
  -asin(22 / 23.75),
  asin(22 / 23.75),
  length.out = 150
)

# Connect the right corner, curved arc, and left corner.
three_point_line <- tibble(
  x = c(
    -47,
    -41.75 + 23.75 * cos(arc_angles),
    -47
  ),
  y = c(
    -22,
    23.75 * sin(arc_angles),
    22
  )
)

shot_plot <- ggplot(
  plot_data,
  aes(x = locationx, y = locationy, color = zone)
) +
  
  # Each point represents one shot.
  # alpha makes points partially transparent.
  geom_point(alpha = 0.4, size = 0.8) +
  
  # Outline the paint.
  annotate(
    "rect",
    xmin = -47, xmax = -28,
    ymin = -8, ymax = 8,
    fill = NA, color = "grey30"
  ) +
  
  # Draw the three-point line.
  geom_path(
    data = three_point_line,
    aes(x = x, y = y),
    inherit.aes = FALSE,
    color = "grey30"
  ) +
  
  # Mark the basket.
  annotate(
    "point",
    x = -41.75, y = 0,
    color = "black", shape = 4, size = 3
  ) +
  
  # Mark half court.
  geom_vline(
    xintercept = 0,
    color = "grey30",
    linetype = "dashed"
  ) +
  
  # Match your drawing: negative y at the top, positive y at the bottom.
  scale_y_reverse() +
  
  labs(
    x = "locationx (feet)",
    y = "locationy (feet)",
    color = "Shot zone"
  ) +
  
  theme_minimal()

# Zoom in to inspect the normal shooting zones.
shot_plot +
  coord_fixed(xlim = c(-47, 0), ylim = c(25, -25)) +
  labs(title = "Shot zones: shooting half")

# Include the other half to see half-court and longer attempts.
shot_plot +
  coord_fixed(xlim = c(-47, 47), ylim = c(25, -25)) +
  labs(title = "Shot zones: full court")

# 4. Season split and leakage-safe shooter features --------------------------

# Keep train intact; create two separate tables.
fit_data <- train %>%
  filter(season_id != "fe055")

validation_data <- train %>%
  filter(season_id == "fe055")

# Use the existing helper instead of repeating its five-fold loop here.
fit_features <- make_oof_features(fit_data)
validation_features <- add_shooter_features(fit_data, validation_data, k = 200)

game_state_levels <- c(
  "HALFCOURT",
  "TRANSITION",
  "DIRECTSECONDCHANCE"
)

# The existing context helper performs the same contest/clock conversions for
# both directions; it does not depend on which season is the reference season.
fit_features <- fit_features %>%
  add_defender_approach() %>%
  add_shooter_movement() %>%
  prepare_reverse_context()
validation_features <- validation_features %>%
  add_defender_approach() %>%
  add_shooter_movement() %>%
  prepare_reverse_context()

shot_type_levels <- sort(unique(fit_features$shottype))
fit_features <- fit_features %>%
  mutate(shottype = factor(shottype, levels = shot_type_levels), zone = factor(zone))
validation_features <- validation_features %>%
  mutate(shottype = factor(shottype, levels = shot_type_levels),
         zone = factor(zone, levels = levels(fit_features$zone)))
stopifnot(identical(fit_features$shot_id, fit_data$shot_id),
          identical(validation_features$shot_id, validation_data$shot_id))

# Stop if any shots have missing categories or unusable distance features.
stopifnot(
  !anyNA(fit_features$contest_group),
  !anyNA(validation_features$contest_group),
  all(is.finite(fit_features$contest_log_distance)),
  all(is.finite(validation_features$contest_log_distance))
)

# 5. Final predictor list and unchanged model settings -----------------------

model_features <- c(
  "zone",
  "shottype",
  "shooter_zone_prob",
  "rim_distance",
  "closestdefdist",
  "closing_speed_1s",
  "closing_speed_last_025s",
  "shooterspeed",
  "dribblesbefore",
  "no_dribbles",
  "shotclock",
  "gamestate",
  "contest_group",
  "contest_log_distance"
)
tree_parameters <- list(
  
  # Predict a probability for a binary outcome.
  objective = "binary:logistic",
  
  # Evaluate predictions using log-loss.
  eval_metric = "logloss",
  
  # Efficient tree-building method.
  tree_method = "hist",
  
  # Each new tree makes a relatively small adjustment.
  eta = 0.03,
  
  # Limit tree depth and complexity.
  max_depth = 4,
  
  # Discourage leaves supported by too little information.
  min_child_weight = 20,
  
  # Use a random 80% of fitting rows for each tree.
  subsample = 0.8,
  
  # Use a random 80% of feature columns for each tree.
  colsample_bytree = 0.8,
  
  # Penalize large leaf adjustments.
  lambda = 5,
  
  # Use two CPU threads.
  nthread = 2,
  
  seed = 2027
)
location_fit_features <- add_precise_location(fit_features)

location_validation_features <- add_precise_location(
  validation_features
)

# Keep all original predictors and add these three.
location_model_features <- unique(c(
  model_features,
  "locationx",
  "locationy",
  "lateral_distance"
))
location_parameters <- tree_parameters
location_parameters$max_depth <- 4L
# These columns were created earlier from closestdefapproach.
approach_columns <- c(
  "def_dist_1s",
  "def_dist_075s",
  "def_dist_05s",
  "def_dist_025s"
)

# Retain all predictors from the precise-location model.
approach_model_features <- unique(c(
  location_model_features,
  approach_columns
))

approach_fit_inputs <- location_fit_features %>%
  select(all_of(approach_model_features))

approach_validation_inputs <- location_validation_features %>%
  select(all_of(approach_model_features))

# Confirm the four distances are numeric, finite, and available.
stopifnot(
  all(vapply(
    approach_fit_inputs[approach_columns],
    is.numeric,
    logical(1)
  )),
  all(vapply(
    approach_validation_inputs[approach_columns],
    is.numeric,
    logical(1)
  )),
  all(is.finite(as.matrix(
    approach_fit_inputs[approach_columns]
  ))),
  all(is.finite(as.matrix(
    approach_validation_inputs[approach_columns]
  ))),
  !anyNA(approach_fit_inputs),
  !anyNA(approach_validation_inputs)
)

x_approach_fit <- sparse.model.matrix(
  ~ . - 1,
  data = approach_fit_inputs
)

x_approach_validation <- sparse.model.matrix(
  ~ . - 1,
  data = approach_validation_inputs
)

# Check column alignment and make sure no shots were dropped.
stopifnot(
  identical(
    colnames(x_approach_fit),
    colnames(x_approach_validation)
  ),
  nrow(x_approach_fit) == nrow(fit_features),
  nrow(x_approach_validation) == nrow(validation_features)
)

d_approach_fit <- xgb.DMatrix(
  x_approach_fit,
  label = fit_features$outcome
)

d_approach_validation <- xgb.DMatrix(
  x_approach_validation,
  label = validation_features$outcome
)


# 6. Fit and validate the final selected model --------------------------------

set.seed(2027)

approach_model <- xgb.train(
  params = location_parameters,
  data = d_approach_fit,
  nrounds = 1500,
  
  evals = list(
    training = d_approach_fit,
    validation = d_approach_validation
  ),
  
  early_stopping_rounds = 50,
  maximize = FALSE,
  print_every_n = 100,
  verbose = 1
)

# Predict using the best iteration selected by early stopping.
approach_predictions <- predict(
  approach_model,
  d_approach_validation
)

stopifnot(
  length(approach_predictions) == nrow(validation_features),
  all(is.finite(approach_predictions)),
  all(approach_predictions >= 0 & approach_predictions <= 1)
)
# Store the chosen round count for a possible reverse-split check.
approach_best_rounds <- as.integer(
  xgb.attr(approach_model, "best_iteration")
) + 1L

print(approach_best_rounds)

# 7. Reverse-season robustness check -----------------------------------------

# Out-of-fold estimates for the new fitting season.
reverse_fit_features <- make_oof_features(validation_data) %>%
  add_defender_approach() %>%
  add_shooter_movement() %>%
  mutate(
    zone = factor(zone),
    shottype = factor(shottype)
  )

# Predictors for the new held-out season.
# Its outcomes are not used to calculate these shooter estimates.
reverse_validation_features <- add_shooter_features(
  reference_data = validation_data,
  target_data = fit_data,
  k = 200
) %>%
  add_defender_approach() %>%
  add_shooter_movement() %>%
  mutate(
    zone = factor(
      zone,
      levels = levels(reverse_fit_features$zone)
    ),
    shottype = factor(
      shottype,
      levels = levels(reverse_fit_features$shottype)
    )
  )

# Check for missing or unrecognized categories.
stopifnot(
  !anyNA(reverse_validation_features$zone),
  !anyNA(reverse_validation_features$shottype)
)
reverse_fit_features <- prepare_reverse_context(
  reverse_fit_features
)

reverse_validation_features <- prepare_reverse_context(
  reverse_validation_features
)
# Use the same predictors and location transformation.
reverse_approach_fit_inputs <- reverse_fit_features %>%
  add_precise_location() %>%
  select(all_of(approach_model_features))

reverse_approach_validation_inputs <- reverse_validation_features %>%
  add_precise_location() %>%
  select(all_of(approach_model_features))

stopifnot(
  !anyNA(reverse_approach_fit_inputs),
  !anyNA(reverse_approach_validation_inputs)
)

x_reverse_approach_fit <- sparse.model.matrix(
  ~ . - 1,
  data = reverse_approach_fit_inputs
)

x_reverse_approach_validation <- sparse.model.matrix(
  ~ . - 1,
  data = reverse_approach_validation_inputs
)

# Ensure consistent columns and preserve every shot.
stopifnot(
  identical(
    colnames(x_reverse_approach_fit),
    colnames(x_reverse_approach_validation)
  ),
  nrow(x_reverse_approach_fit) == nrow(reverse_fit_features),
  nrow(x_reverse_approach_validation) ==
    nrow(reverse_validation_features)
)

d_reverse_approach_fit <- xgb.DMatrix(
  x_reverse_approach_fit,
  label = reverse_fit_features$outcome
)

d_reverse_approach_validation <- xgb.DMatrix(
  x_reverse_approach_validation,
  label = reverse_validation_features$outcome
)

set.seed(2027)

reverse_approach_model <- xgb.train(
  params = location_parameters,
  data = d_reverse_approach_fit,
  
  # Fix this using the original split's selected round count.
  nrounds = approach_best_rounds,
  
  evals = list(
    training = d_reverse_approach_fit,
    validation = d_reverse_approach_validation
  ),
  
  # No early stopping on the reverse check.
  print_every_n = 100,
  verbose = 1
)

reverse_approach_predictions <- predict(
  reverse_approach_model,
  d_reverse_approach_validation
)

stopifnot(
  length(reverse_approach_predictions) ==
    nrow(reverse_validation_features),
  all(is.finite(reverse_approach_predictions)),
  all(
    reverse_approach_predictions >= 0 &
      reverse_approach_predictions <= 1
  )
)

# One compact table replaces the repeated trial-by-trial comparison tables.
validation_scores <- tibble(
  direction = rep(c("Original", "Reversed"), each = 3),
  model = rep(c("Zone average", "Shooter + zone, k = 200",
                "XGBoost: location + full approach"), times = 2),
  log_loss = c(
    log_loss(validation_features$outcome, validation_features$zone_prob),
    log_loss(validation_features$outcome, validation_features$shooter_zone_prob),
    log_loss(validation_features$outcome, approach_predictions),
    log_loss(reverse_validation_features$outcome, reverse_validation_features$zone_prob),
    log_loss(reverse_validation_features$outcome, reverse_validation_features$shooter_zone_prob),
    log_loss(reverse_validation_features$outcome, reverse_approach_predictions)
  )
)
print(as.data.frame(validation_scores), digits = 7, row.names = FALSE)

# 8. Refit the SAME model on every labeled training shot ----------------------
# Each training shot still receives out-of-fold shooter statistics. Test shots
# receive shooter statistics learned from the complete labeled training data.
full_fit_features <- make_oof_features(train) %>%
  add_defender_approach() %>%
  add_shooter_movement() %>%
  prepare_reverse_context() %>%
  add_precise_location()

test_features <- add_shooter_features(train, test, k = 200) %>%
  add_defender_approach() %>%
  add_shooter_movement() %>%
  prepare_reverse_context() %>%
  add_precise_location()

full_fit_features <- full_fit_features %>%
  mutate(zone = factor(zone), shottype = factor(shottype))
test_features <- test_features %>%
  mutate(zone = factor(zone, levels = levels(full_fit_features$zone)),
         shottype = factor(shottype, levels = levels(full_fit_features$shottype)))

full_fit_inputs <- full_fit_features %>% select(all_of(approach_model_features))
test_inputs <- test_features %>% select(all_of(approach_model_features))
stopifnot(!anyNA(full_fit_inputs), !anyNA(test_inputs),
          identical(full_fit_features$shot_id, train$shot_id),
          identical(test_features$shot_id, test$shot_id))

x_full_fit <- sparse.model.matrix(~ . - 1, data = full_fit_inputs)
x_test <- sparse.model.matrix(~ . - 1, data = test_inputs)
stopifnot(identical(colnames(x_full_fit), colnames(x_test)),
          nrow(x_full_fit) == nrow(train), nrow(x_test) == nrow(test))

d_full_fit <- xgb.DMatrix(x_full_fit, label = full_fit_features$outcome)
d_test <- xgb.DMatrix(x_test)

set.seed(2027)
final_model <- xgb.train(
  params = location_parameters,
  data = d_full_fit,
  # Reuse the round count selected by validation, not the training score.
  nrounds = approach_best_rounds,
  verbose = 0
)
test_predictions <- predict(final_model, d_test)
stopifnot(length(test_predictions) == nrow(test),
          all(is.finite(test_predictions)),
          all(test_predictions >= 0 & test_predictions <= 1))

# Feature importance is available for the write-up; no new fitting is needed.
final_feature_importance <- xgb.importance(model = final_model)
print(head(final_feature_importance, 20))

# 9. Fill submission.csv without changing columns or template row order -------
submission_path <- file.path(data_path, "submission.csv")
submission <- read_csv(
  submission_path,
  col_types = cols(shot_id = col_character(), make_prob = col_double())
)
stopifnot(identical(names(submission), c("shot_id", "make_prob")),
          nrow(submission) == nrow(test), !anyNA(submission$shot_id),
          !anyDuplicated(submission$shot_id),
          setequal(submission$shot_id, test$shot_id))

submission_ids <- submission$shot_id
submission$make_prob <- test_predictions[match(submission$shot_id, test$shot_id)]
stopifnot(identical(submission$shot_id, submission_ids),
          all(is.finite(submission$make_prob)),
          all(submission$make_prob >= 0 & submission$make_prob <= 1))

# Default: write the requested submission. Setting this option to FALSE permits
# a dry run that checks the complete workflow without changing the CSV.
if (isTRUE(getOption("analyst.write_submission", TRUE))) {
  write_csv(submission, submission_path)
  message("Wrote ", nrow(submission), " predictions to submission.csv.")
} else {
  message("Dry run: submission predictions are validated and held in memory.")
}

