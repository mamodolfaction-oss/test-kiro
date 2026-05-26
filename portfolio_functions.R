# =============================================================================
# Portfolio Optimization Functions for LQ45 Stock Analysis
# =============================================================================

library(quadprog)

#' Calculate portfolio return and risk
#' @param weights Vector of portfolio weights
#' @param returns Matrix of asset returns
#' @return List with portfolio return and risk
calculate_portfolio_metrics <- function(weights, returns) {
  port_return <- sum(weights * colMeans(returns)) * 252  # Annualized
  cov_matrix <- cov(returns) * 252  # Annualized
  port_risk <- sqrt(t(weights) %*% cov_matrix %*% weights)
  sharpe <- port_return / port_risk
  
  list(
    return = port_return,
    risk = as.numeric(port_risk),
    sharpe = as.numeric(sharpe)
  )
}

#' Optimize portfolio for minimum variance
#' @param returns Matrix of daily returns
#' @return Optimal weights
optimize_min_variance <- function(returns) {
  n <- ncol(returns)
  cov_matrix <- cov(returns) * 252
  
  # Quadratic programming: min 0.5 * w' * Sigma * w
  # Subject to: sum(w) = 1, w >= 0
  Dmat <- 2 * cov_matrix
  dvec <- rep(0, n)
  
  # Constraints: sum(w) = 1 and w >= 0
  Amat <- cbind(rep(1, n), diag(n))
  bvec <- c(1, rep(0, n))
  meq <- 1
  
  tryCatch({
    sol <- solve.QP(Dmat, dvec, Amat, bvec, meq)
    weights <- sol$solution
    weights[weights < 1e-6] <- 0
    weights <- weights / sum(weights)
    return(weights)
  }, error = function(e) {
    return(rep(1/n, n))  # Equal weight fallback
  })
}

#' Optimize portfolio for maximum Sharpe ratio
#' @param returns Matrix of daily returns
#' @param rf Risk-free rate (annual)
#' @return Optimal weights
optimize_max_sharpe <- function(returns, rf = 0.05) {

  n <- ncol(returns)
  mu <- colMeans(returns) * 252
  cov_matrix <- cov(returns) * 252
  
  # Excess returns
  excess_mu <- mu - rf
  
  # Solve: min w'Σw subject to w'(μ-rf) = 1, w >= 0

  Dmat <- 2 * cov_matrix
  dvec <- rep(0, n)
  
  Amat <- cbind(excess_mu, diag(n))
  bvec <- c(1, rep(0, n))
  meq <- 1
  
  tryCatch({
    sol <- solve.QP(Dmat, dvec, Amat, bvec, meq)
    weights <- sol$solution
    weights[weights < 1e-6] <- 0
    weights <- weights / sum(weights)
    return(weights)
  }, error = function(e) {
    return(rep(1/n, n))  # Equal weight fallback
  })
}

#' Generate efficient frontier points
#' @param returns Matrix of daily returns
#' @param n_points Number of points on the frontier
#' @return Data frame with risk and return
generate_efficient_frontier <- function(returns, n_points = 50) {
  n <- ncol(returns)
  mu <- colMeans(returns) * 252
  cov_matrix <- cov(returns) * 252
  
  # Get range of target returns
  min_ret <- min(mu)
  max_ret <- max(mu)
  target_returns <- seq(min_ret * 0.8, max_ret * 1.2, length.out = n_points)
  
  frontier <- data.frame(Risk = numeric(0), Return = numeric(0))
  
  for (target in target_returns) {
    # min w'Σw subject to: w'μ = target, sum(w) = 1, w >= 0
    Dmat <- 2 * cov_matrix
    dvec <- rep(0, n)
    
    Amat <- cbind(mu, rep(1, n), diag(n))
    bvec <- c(target, 1, rep(0, n))
    meq <- 2
    
    tryCatch({
      sol <- solve.QP(Dmat, dvec, Amat, bvec, meq)
      risk <- sqrt(t(sol$solution) %*% cov_matrix %*% sol$solution)
      frontier <- rbind(frontier, data.frame(Risk = as.numeric(risk), Return = target))
    }, error = function(e) {
      # Skip infeasible points
    })
  }
  
  return(frontier)
}

#' Calculate Value at Risk (VaR)
#' @param returns Portfolio returns
#' @param confidence Confidence level
#' @param investment Investment amount
#' @return VaR value
calculate_var <- function(returns, confidence = 0.95, investment = 1000000) {
  var_pct <- quantile(returns, 1 - confidence)
  var_amount <- abs(var_pct) * investment
  list(
    var_pct = as.numeric(var_pct),
    var_amount = as.numeric(var_amount)
  )
}

#' Calculate Conditional VaR (CVaR / Expected Shortfall)
#' @param returns Portfolio returns
#' @param confidence Confidence level
#' @param investment Investment amount
#' @return CVaR value
calculate_cvar <- function(returns, confidence = 0.95, investment = 1000000) {
  var_pct <- quantile(returns, 1 - confidence)
  cvar_pct <- mean(returns[returns <= var_pct])
  cvar_amount <- abs(cvar_pct) * investment
  list(
    cvar_pct = as.numeric(cvar_pct),
    cvar_amount = as.numeric(cvar_amount)
  )
}

#' Calculate Maximum Drawdown
#' @param cum_returns Cumulative returns series
#' @return Maximum drawdown value
calculate_max_drawdown <- function(cum_returns) {
  peak <- cummax(cum_returns)
  drawdown <- (cum_returns - peak) / peak
  return(min(drawdown))
}

#' List of LQ45 stocks (as of recent composition)
get_lq45_stocks <- function() {
  stocks <- c(
    "ACES.JK", "ADRO.JK", "AKRA.JK", "AMRT.JK", "ANTM.JK",
    "ASII.JK", "BBCA.JK", "BBNI.JK", "BBRI.JK", "BBTN.JK",
    "BMRI.JK", "BRPT.JK", "BUKA.JK", "CPIN.JK", "EMTK.JK",
    "ERAA.JK", "ESSA.JK", "EXCL.JK", "GGRM.JK", "GOTO.JK",
    "HRUM.JK", "ICBP.JK", "INCO.JK", "INDF.JK", "INKP.JK",
    "ITMG.JK", "JPFA.JK", "KLBF.JK", "MAPI.JK", "MDKA.JK",
    "MEDC.JK", "MIKA.JK", "MNCN.JK", "PGAS.JK", "PTBA.JK",
    "SMGR.JK", "TBIG.JK", "TINS.JK", "TLKM.JK", "TOWR.JK",
    "TPIA.JK", "UNTR.JK", "UNVR.JK", "WMUU.JK", "WIKA.JK"
  )
  return(stocks)
}
