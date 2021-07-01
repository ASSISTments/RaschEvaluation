data {
  int N; // number of observations
  int S; // number of students
  int Q; // number of unique questions
  int C; // number of unique score types
  int N_OOS; // number of hold-out
  
  int<lower = 1, upper = S> student[N];
  int<lower = 1, upper = Q> question[N];
  int<lower = 1, upper = C> grade[N];
  
  int<lower = 1, upper = S> student_oos[N_OOS];
  int<lower = 1, upper = Q> question_oos[N_OOS];
}
parameters {
  vector[S] student_quality;
  vector[Q] question_difficulty;
  ordered[C-1] cut_points;
  real<lower = 0> sigma_student;
  real<lower = 0> sigma_question;
}
model {
  // priors
  student_quality ~ normal(0, sigma_student);
  question_difficulty ~ normal(0, sigma_question);
  cut_points ~ normal(0, 2);

  // likelihood
  grade ~ ordered_logistic(student_quality[student] - question_difficulty[question], cut_points);
}
generated quantities {
  int<lower=1, upper=C> y_pred[N_OOS];
  
  for (n in 1:N_OOS)
    y_pred[n] = ordered_logistic_rng(student_quality[student_oos[n]] - question_difficulty[question_oos[n]], cut_points);
}
