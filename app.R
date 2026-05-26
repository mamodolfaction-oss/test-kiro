# =============================================================================
# R Shiny App: Analisis Portofolio Saham LQ45
# =============================================================================

library(shiny)
library(shinydashboard)
library(quantmod)
library(PerformanceAnalytics)
library(plotly)
library(DT)
library(dplyr)
library(tidyr)
library(ggplot2)
library(corrplot)

source("portfolio_functions.R")

# =============================================================================
# UI
# =============================================================================

ui <- dashboardPage(
  dashboardHeader(title = "Analisis Portofolio LQ45"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Data & Saham", tabName = "data", icon = icon("database")),
      menuItem("Analisis Individual", tabName = "individual", icon = icon("chart-line")),
      menuItem("Korelasi", tabName = "correlation", icon = icon("project-diagram")),
      menuItem("Optimasi Portofolio", tabName = "optimize", icon = icon("balance-scale")),
      menuItem("Kinerja Portofolio", tabName = "performance", icon = icon("tachometer-alt")),
      menuItem("Manajemen Risiko", tabName = "risk", icon = icon("exclamation-triangle"))
    )
  ),

  
  dashboardBody(
    tabItems(
      # --- Tab 1: Data & Saham ---
      tabItem(tabName = "data",
        fluidRow(
          box(title = "Pengaturan Data", status = "primary", solidHeader = TRUE, width = 4,
            selectInput("selected_stocks", "Pilih Saham LQ45:",
              choices = get_lq45_stocks(),
              selected = c("BBCA.JK", "BBRI.JK", "TLKM.JK", "ASII.JK", "UNVR.JK"),
              multiple = TRUE
            ),
            dateRangeInput("date_range", "Periode Analisis:",
              start = Sys.Date() - 365,
              end = Sys.Date(),
              format = "yyyy-mm-dd"
            ),
            actionButton("fetch_data", "Ambil Data", 
              class = "btn-primary", icon = icon("download")),
            br(), br(),
            verbatimTextOutput("data_status")
          ),
          box(title = "Ringkasan Data", status = "info", solidHeader = TRUE, width = 8,
            DTOutput("data_summary_table")
          )
        ),
        fluidRow(
          box(title = "Harga Saham", status = "success", solidHeader = TRUE, width = 12,
            plotlyOutput("price_chart", height = "400px")
          )
        )
      ),

      
      # --- Tab 2: Analisis Individual ---
      tabItem(tabName = "individual",
        fluidRow(
          box(title = "Statistik Deskriptif", status = "primary", solidHeader = TRUE, width = 12,
            DTOutput("desc_stats_table")
          )
        ),
        fluidRow(
          box(title = "Distribusi Return", status = "info", solidHeader = TRUE, width = 6,
            plotlyOutput("return_dist_chart", height = "350px")
          ),
          box(title = "Return Kumulatif", status = "success", solidHeader = TRUE, width = 6,
            plotlyOutput("cumulative_return_chart", height = "350px")
          )
        )
      ),
      
      # --- Tab 3: Korelasi ---
      tabItem(tabName = "correlation",
        fluidRow(
          box(title = "Matriks Korelasi", status = "primary", solidHeader = TRUE, width = 7,
            plotOutput("correlation_matrix", height = "500px")
          ),
          box(title = "Tabel Korelasi", status = "info", solidHeader = TRUE, width = 5,
            DTOutput("correlation_table")
          )
        )
      ),

      
      # --- Tab 4: Optimasi Portofolio ---
      tabItem(tabName = "optimize",
        fluidRow(
          box(title = "Metode Optimasi", status = "primary", solidHeader = TRUE, width = 4,
            radioButtons("opt_method", "Pilih Metode:",
              choices = list(
                "Minimum Variance" = "min_var",
                "Maximum Sharpe Ratio" = "max_sharpe",
                "Equal Weight" = "equal_weight"
              ),
              selected = "max_sharpe"
            ),
            numericInput("risk_free_rate", "Risk-Free Rate (%):", 
              value = 5, min = 0, max = 20, step = 0.5),
            actionButton("run_optimization", "Jalankan Optimasi",
              class = "btn-success", icon = icon("cogs")),
            br(), br(),
            h4("Hasil Portofolio Optimal:"),
            verbatimTextOutput("opt_results")
          ),
          box(title = "Alokasi Bobot Optimal", status = "success", solidHeader = TRUE, width = 8,
            plotlyOutput("weights_chart", height = "300px"),
            br(),
            DTOutput("weights_table")
          )
        ),
        fluidRow(
          box(title = "Efficient Frontier", status = "warning", solidHeader = TRUE, width = 12,
            plotlyOutput("efficient_frontier_chart", height = "450px")
          )
        )
      ),

      
      # --- Tab 5: Kinerja Portofolio ---
      tabItem(tabName = "performance",
        fluidRow(
          box(title = "Pengaturan Portofolio", status = "primary", solidHeader = TRUE, width = 4,
            numericInput("investment_amount", "Jumlah Investasi (Rp):",
              value = 100000000, min = 1000000, step = 1000000),
            uiOutput("manual_weights_ui"),
            actionButton("calc_performance", "Hitung Kinerja",
              class = "btn-primary", icon = icon("calculator"))
          ),
          box(title = "Metrik Kinerja", status = "info", solidHeader = TRUE, width = 8,
            fluidRow(
              valueBoxOutput("annual_return_box", width = 4),
              valueBoxOutput("annual_risk_box", width = 4),
              valueBoxOutput("sharpe_ratio_box", width = 4)
            )
          )
        ),
        fluidRow(
          box(title = "Pertumbuhan Portofolio", status = "success", solidHeader = TRUE, width = 12,
            plotlyOutput("portfolio_growth_chart", height = "400px")
          )
        )
      ),
      
      # --- Tab 6: Manajemen Risiko ---
      tabItem(tabName = "risk",
        fluidRow(
          box(title = "Pengaturan Risiko", status = "primary", solidHeader = TRUE, width = 4,
            sliderInput("var_confidence", "Confidence Level VaR:", 
              min = 0.90, max = 0.99, value = 0.95, step = 0.01),
            numericInput("var_investment", "Nilai Investasi (Rp):",
              value = 100000000, min = 1000000, step = 1000000),
            actionButton("calc_risk", "Hitung Risiko",
              class = "btn-danger", icon = icon("exclamation-triangle"))
          ),
          box(title = "Metrik Risiko", status = "danger", solidHeader = TRUE, width = 8,
            fluidRow(
              valueBoxOutput("var_box", width = 4),
              valueBoxOutput("cvar_box", width = 4),
              valueBoxOutput("max_dd_box", width = 4)
            ),
            br(),
            verbatimTextOutput("risk_summary")
          )
        ),
        fluidRow(
          box(title = "Distribusi Return Portofolio", status = "warning", solidHeader = TRUE, width = 6,
            plotlyOutput("risk_dist_chart", height = "350px")
          ),
          box(title = "Drawdown", status = "danger", solidHeader = TRUE, width = 6,
            plotlyOutput("drawdown_chart", height = "350px")
          )
        )
      )
    )
  )
)


# =============================================================================
# SERVER
# =============================================================================

server <- function(input, output, session) {
  
  # Reactive values
  rv <- reactiveValues(
    stock_data = NULL,
    returns = NULL,
    optimal_weights = NULL
  )
  
  # --- Fetch stock data ---
  observeEvent(input$fetch_data, {
    req(length(input$selected_stocks) >= 2)
    
    withProgress(message = "Mengambil data saham...", value = 0, {
      stock_list <- list()
      n_stocks <- length(input$selected_stocks)
      
      for (i in seq_along(input$selected_stocks)) {
        ticker <- input$selected_stocks[i]
        incProgress(1/n_stocks, detail = paste("Downloading", ticker))
        
        tryCatch({
          data <- getSymbols(ticker, src = "yahoo",
            from = input$date_range[1], to = input$date_range[2],
            auto.assign = FALSE)
          stock_list[[ticker]] <- Ad(data)  # Adjusted close
        }, error = function(e) {
          showNotification(paste("Gagal mengambil data:", ticker), type = "error")
        })
      }
      
      if (length(stock_list) >= 2) {
        # Merge all prices
        prices <- do.call(merge, stock_list)
        prices <- na.omit(prices)
        colnames(prices) <- names(stock_list)
        
        # Calculate returns
        returns <- na.omit(Return.calculate(prices, method = "log"))
        
        rv$stock_data <- prices
        rv$returns <- returns
        
        showNotification("Data berhasil diambil!", type = "message")
      }
    })
  })

  
  # --- Data status ---
  output$data_status <- renderText({
    if (is.null(rv$stock_data)) {
      "Belum ada data. Pilih saham dan klik 'Ambil Data'."
    } else {
      paste0("Data tersedia: ", ncol(rv$stock_data), " saham, ",
             nrow(rv$stock_data), " hari perdagangan\n",
             "Periode: ", index(rv$stock_data)[1], " s/d ", 
             tail(index(rv$stock_data), 1))
    }
  })
  
  # --- Data summary table ---
  output$data_summary_table <- renderDT({
    req(rv$stock_data)
    prices <- rv$stock_data
    
    summary_df <- data.frame(
      Saham = colnames(prices),
      `Harga Terakhir` = as.numeric(tail(prices, 1)),
      `Harga Tertinggi` = apply(prices, 2, max),
      `Harga Terendah` = apply(prices, 2, min),
      `Rata-rata` = colMeans(prices),
      check.names = FALSE
    )
    
    datatable(summary_df, options = list(pageLength = 10, scrollX = TRUE)) %>%
      formatRound(columns = 2:5, digits = 0)
  })
  
  # --- Price chart ---
  output$price_chart <- renderPlotly({
    req(rv$stock_data)
    prices <- rv$stock_data
    
    # Normalize to 100
    normalized <- sweep(prices, 2, as.numeric(prices[1,]), "/") * 100
    
    df <- data.frame(
      Date = index(normalized),
      coredata(normalized)
    )
    df_long <- pivot_longer(df, -Date, names_to = "Saham", values_to = "Harga")
    
    plot_ly(df_long, x = ~Date, y = ~Harga, color = ~Saham, type = "scatter", mode = "lines") %>%
      layout(title = "Pergerakan Harga (Normalized = 100)",
             xaxis = list(title = "Tanggal"),
             yaxis = list(title = "Harga Relatif"))
  })

  
  # --- Descriptive statistics ---
  output$desc_stats_table <- renderDT({
    req(rv$returns)
    returns <- rv$returns
    
    stats_df <- data.frame(
      Saham = colnames(returns),
      `Return Harian (%)` = colMeans(returns) * 100,
      `Return Tahunan (%)` = colMeans(returns) * 252 * 100,
      `Volatilitas Harian (%)` = apply(returns, 2, sd) * 100,
      `Volatilitas Tahunan (%)` = apply(returns, 2, sd) * sqrt(252) * 100,
      Skewness = apply(returns, 2, function(x) moments::skewness(as.numeric(x))),
      Kurtosis = apply(returns, 2, function(x) moments::kurtosis(as.numeric(x))),
      `Sharpe Ratio` = (colMeans(returns) * 252) / (apply(returns, 2, sd) * sqrt(252)),
      check.names = FALSE
    )
    
    datatable(stats_df, options = list(pageLength = 15, scrollX = TRUE)) %>%
      formatRound(columns = 2:8, digits = 4)
  })
  
  # --- Return distribution ---
  output$return_dist_chart <- renderPlotly({
    req(rv$returns)
    returns <- rv$returns
    
    df <- data.frame(coredata(returns))
    df_long <- pivot_longer(df, everything(), names_to = "Saham", values_to = "Return")
    
    plot_ly(df_long, x = ~Return, color = ~Saham, type = "histogram", 
            opacity = 0.6, nbinsx = 50) %>%
      layout(title = "Distribusi Return Harian",
             xaxis = list(title = "Return"),
             yaxis = list(title = "Frekuensi"),
             barmode = "overlay")
  })
  
  # --- Cumulative return chart ---
  output$cumulative_return_chart <- renderPlotly({
    req(rv$returns)
    cum_returns <- cumprod(1 + rv$returns) - 1
    
    df <- data.frame(
      Date = index(cum_returns),
      coredata(cum_returns)
    )
    df_long <- pivot_longer(df, -Date, names_to = "Saham", values_to = "CumReturn")
    
    plot_ly(df_long, x = ~Date, y = ~CumReturn * 100, color = ~Saham, 
            type = "scatter", mode = "lines") %>%
      layout(title = "Return Kumulatif (%)",
             xaxis = list(title = "Tanggal"),
             yaxis = list(title = "Return Kumulatif (%)"))
  })

  
  # --- Correlation matrix ---
  output$correlation_matrix <- renderPlot({
    req(rv$returns)
    cor_matrix <- cor(rv$returns)
    corrplot(cor_matrix, method = "color", type = "upper", 
             tl.col = "black", tl.srt = 45, 
             addCoef.col = "black", number.cex = 0.7,
             title = "Matriks Korelasi Return Saham",
             mar = c(0, 0, 2, 0))
  })
  
  # --- Correlation table ---
  output$correlation_table <- renderDT({
    req(rv$returns)
    cor_matrix <- round(cor(rv$returns), 4)
    cor_df <- as.data.frame(cor_matrix)
    cor_df$Saham <- rownames(cor_df)
    cor_df <- cor_df[, c("Saham", colnames(cor_matrix))]
    
    datatable(cor_df, options = list(pageLength = 15, scrollX = TRUE)) %>%
      formatRound(columns = 2:ncol(cor_df), digits = 4)
  })
  
  # --- Portfolio Optimization ---
  observeEvent(input$run_optimization, {
    req(rv$returns)
    
    returns_mat <- as.matrix(rv$returns)
    rf <- input$risk_free_rate / 100
    
    weights <- switch(input$opt_method,
      "min_var" = optimize_min_variance(returns_mat),
      "max_sharpe" = optimize_max_sharpe(returns_mat, rf),
      "equal_weight" = rep(1/ncol(returns_mat), ncol(returns_mat))
    )
    
    rv$optimal_weights <- weights
  })
  
  # --- Optimization results ---
  output$opt_results <- renderText({
    req(rv$optimal_weights, rv$returns)
    
    metrics <- calculate_portfolio_metrics(rv$optimal_weights, as.matrix(rv$returns))
    
    paste0(
      "Expected Return (tahunan): ", round(metrics$return * 100, 2), "%\n",
      "Risiko/Volatilitas (tahunan): ", round(metrics$risk * 100, 2), "%\n",
      "Sharpe Ratio: ", round(metrics$sharpe, 4)
    )
  })

  
  # --- Weights chart ---
  output$weights_chart <- renderPlotly({
    req(rv$optimal_weights)
    
    df <- data.frame(
      Saham = colnames(rv$returns),
      Bobot = rv$optimal_weights
    )
    df <- df[df$Bobot > 0.001, ]  # Filter negligible weights
    
    plot_ly(df, labels = ~Saham, values = ~Bobot, type = "pie",
            textinfo = "label+percent") %>%
      layout(title = "Alokasi Bobot Portofolio Optimal")
  })
  
  # --- Weights table ---
  output$weights_table <- renderDT({
    req(rv$optimal_weights)
    
    df <- data.frame(
      Saham = colnames(rv$returns),
      `Bobot (%)` = round(rv$optimal_weights * 100, 2),
      check.names = FALSE
    )
    df <- df[order(-df$`Bobot (%)`), ]
    
    datatable(df, options = list(pageLength = 10)) %>%
      formatRound(columns = 2, digits = 2)
  })
  
  # --- Efficient Frontier ---
  output$efficient_frontier_chart <- renderPlotly({
    req(rv$returns)
    
    returns_mat <- as.matrix(rv$returns)
    frontier <- generate_efficient_frontier(returns_mat)
    
    # Individual stock points
    mu <- colMeans(returns_mat) * 252
    sigma <- apply(returns_mat, 2, sd) * sqrt(252)
    stocks_df <- data.frame(Risk = sigma, Return = mu, Saham = colnames(returns_mat))
    
    p <- plot_ly() %>%
      add_trace(data = frontier, x = ~Risk * 100, y = ~Return * 100, 
                type = "scatter", mode = "lines",
                name = "Efficient Frontier",
                line = list(color = "blue", width = 2)) %>%
      add_trace(data = stocks_df, x = ~Risk * 100, y = ~Return * 100,
                type = "scatter", mode = "markers+text",
                text = ~Saham, textposition = "top center",
                name = "Saham Individual",
                marker = list(size = 10, color = "red"))
    
    # Add optimal portfolio point if available
    if (!is.null(rv$optimal_weights)) {
      opt_metrics <- calculate_portfolio_metrics(rv$optimal_weights, returns_mat)
      p <- p %>% add_trace(
        x = opt_metrics$risk * 100, y = opt_metrics$return * 100,
        type = "scatter", mode = "markers",
        name = "Portofolio Optimal",
        marker = list(size = 15, color = "green", symbol = "star")
      )
    }
    
    p %>% layout(
      title = "Efficient Frontier",
      xaxis = list(title = "Risiko / Volatilitas (%)"),
      yaxis = list(title = "Expected Return (%)")
    )
  })

  
  # --- Manual weights UI ---
  output$manual_weights_ui <- renderUI({
    req(rv$returns)
    n <- ncol(rv$returns)
    stock_names <- colnames(rv$returns)
    
    weight_inputs <- lapply(seq_len(n), function(i) {
      numericInput(paste0("weight_", i), 
                   paste0(stock_names[i], " (%):"),
                   value = round(100/n, 1), min = 0, max = 100, step = 1)
    })
    
    tagList(
      h4("Bobot Manual (%)"),
      weight_inputs,
      helpText("Total bobot harus = 100%")
    )
  })
  
  # --- Performance metrics ---
  get_portfolio_returns <- reactive({
    req(rv$returns)
    
    n <- ncol(rv$returns)
    weights <- if (!is.null(rv$optimal_weights)) {
      rv$optimal_weights
    } else {
      rep(1/n, n)
    }
    
    port_returns <- as.matrix(rv$returns) %*% weights
    xts(port_returns, order.by = index(rv$returns))
  })
  
  output$annual_return_box <- renderValueBox({
    req(rv$returns)
    port_ret <- get_portfolio_returns()
    ann_return <- mean(port_ret) * 252 * 100
    
    valueBox(
      paste0(round(ann_return, 2), "%"),
      "Return Tahunan",
      icon = icon("arrow-up"),
      color = if (ann_return >= 0) "green" else "red"
    )
  })
  
  output$annual_risk_box <- renderValueBox({
    req(rv$returns)
    port_ret <- get_portfolio_returns()
    ann_risk <- sd(port_ret) * sqrt(252) * 100
    
    valueBox(
      paste0(round(ann_risk, 2), "%"),
      "Volatilitas Tahunan",
      icon = icon("chart-bar"),
      color = "yellow"
    )
  })
  
  output$sharpe_ratio_box <- renderValueBox({
    req(rv$returns)
    port_ret <- get_portfolio_returns()
    rf <- input$risk_free_rate / 100
    sharpe <- (mean(port_ret) * 252 - rf) / (sd(port_ret) * sqrt(252))
    
    valueBox(
      round(sharpe, 4),
      "Sharpe Ratio",
      icon = icon("trophy"),
      color = if (sharpe >= 1) "green" else if (sharpe >= 0) "yellow" else "red"
    )
  })

  
  # --- Portfolio growth chart ---
  output$portfolio_growth_chart <- renderPlotly({
    req(rv$returns)
    port_ret <- get_portfolio_returns()
    investment <- input$investment_amount
    
    growth <- cumprod(1 + port_ret) * investment
    
    df <- data.frame(
      Date = index(growth),
      Value = as.numeric(growth)
    )
    
    plot_ly(df, x = ~Date, y = ~Value, type = "scatter", mode = "lines",
            fill = "tozeroy", fillcolor = "rgba(0,100,200,0.1)",
            line = list(color = "blue")) %>%
      layout(title = paste0("Pertumbuhan Portofolio (Investasi Awal: Rp ", 
                            format(investment, big.mark = ".", decimal.mark = ","), ")"),
             xaxis = list(title = "Tanggal"),
             yaxis = list(title = "Nilai Portofolio (Rp)",
                         tickformat = ",.0f"))
  })
  
  # --- Risk Management ---
  observeEvent(input$calc_risk, {
    # Trigger recalculation
  })
  
  output$var_box <- renderValueBox({
    req(rv$returns)
    input$calc_risk  # Dependency
    
    port_ret <- get_portfolio_returns()
    var_result <- calculate_var(as.numeric(port_ret), input$var_confidence, input$var_investment)
    
    valueBox(
      paste0("Rp ", format(round(var_result$var_amount), big.mark = ".")),
      paste0("Value at Risk (", input$var_confidence * 100, "%)"),
      icon = icon("exclamation-circle"),
      color = "red"
    )
  })
  
  output$cvar_box <- renderValueBox({
    req(rv$returns)
    input$calc_risk
    
    port_ret <- get_portfolio_returns()
    cvar_result <- calculate_cvar(as.numeric(port_ret), input$var_confidence, input$var_investment)
    
    valueBox(
      paste0("Rp ", format(round(cvar_result$cvar_amount), big.mark = ".")),
      paste0("CVaR / Expected Shortfall"),
      icon = icon("exclamation-triangle"),
      color = "red"
    )
  })
  
  output$max_dd_box <- renderValueBox({
    req(rv$returns)
    input$calc_risk
    
    port_ret <- get_portfolio_returns()
    cum_ret <- cumprod(1 + as.numeric(port_ret))
    max_dd <- calculate_max_drawdown(cum_ret)
    
    valueBox(
      paste0(round(max_dd * 100, 2), "%"),
      "Maximum Drawdown",
      icon = icon("arrow-down"),
      color = "orange"
    )
  })

  
  # --- Risk summary ---
  output$risk_summary <- renderText({
    req(rv$returns)
    input$calc_risk
    
    port_ret <- get_portfolio_returns()
    ret_vec <- as.numeric(port_ret)
    
    var_result <- calculate_var(ret_vec, input$var_confidence, input$var_investment)
    cvar_result <- calculate_cvar(ret_vec, input$var_confidence, input$var_investment)
    cum_ret <- cumprod(1 + ret_vec)
    max_dd <- calculate_max_drawdown(cum_ret)
    
    paste0(
      "=== Ringkasan Manajemen Risiko ===\n\n",
      "Confidence Level: ", input$var_confidence * 100, "%\n",
      "Nilai Investasi: Rp ", format(input$var_investment, big.mark = "."), "\n\n",
      "VaR (1-hari): ", round(var_result$var_pct * 100, 4), "% = Rp ",
        format(round(var_result$var_amount), big.mark = "."), "\n",
      "CVaR (1-hari): ", round(cvar_result$cvar_pct * 100, 4), "% = Rp ",
        format(round(cvar_result$cvar_amount), big.mark = "."), "\n",
      "Maximum Drawdown: ", round(max_dd * 100, 2), "%\n\n",
      "Interpretasi:\n",
      "- Dengan confidence ", input$var_confidence * 100, "%, kerugian harian\n",
      "  tidak akan melebihi Rp ", format(round(var_result$var_amount), big.mark = "."), "\n",
      "- Jika kerugian melebihi VaR, rata-rata kerugian\n",
      "  diperkirakan sebesar Rp ", format(round(cvar_result$cvar_amount), big.mark = ".")
    )
  })
  
  # --- Risk distribution chart ---
  output$risk_dist_chart <- renderPlotly({
    req(rv$returns)
    
    port_ret <- as.numeric(get_portfolio_returns())
    var_val <- quantile(port_ret, 1 - input$var_confidence)
    
    df <- data.frame(Return = port_ret)
    
    plot_ly(df, x = ~Return * 100, type = "histogram", nbinsx = 50,
            marker = list(color = "steelblue")) %>%
      add_segments(x = var_val * 100, xend = var_val * 100, y = 0, yend = 20,
                   line = list(color = "red", width = 2, dash = "dash"),
                   name = paste0("VaR ", input$var_confidence * 100, "%")) %>%
      layout(title = "Distribusi Return Portofolio Harian",
             xaxis = list(title = "Return (%)"),
             yaxis = list(title = "Frekuensi"))
  })
  
  # --- Drawdown chart ---
  output$drawdown_chart <- renderPlotly({
    req(rv$returns)
    
    port_ret <- get_portfolio_returns()
    cum_ret <- cumprod(1 + port_ret)
    peak <- cummax(cum_ret)
    drawdown <- (cum_ret - peak) / peak
    
    df <- data.frame(
      Date = index(drawdown),
      Drawdown = as.numeric(drawdown) * 100
    )
    
    plot_ly(df, x = ~Date, y = ~Drawdown, type = "scatter", mode = "lines",
            fill = "tozeroy", fillcolor = "rgba(255,0,0,0.2)",
            line = list(color = "red")) %>%
      layout(title = "Drawdown Portofolio",
             xaxis = list(title = "Tanggal"),
             yaxis = list(title = "Drawdown (%)"))
  })
}

# =============================================================================
# Run App
# =============================================================================

shinyApp(ui = ui, server = server)
