#Mean Absolute Deviation (MAD) vs Mean Variance optimization for stocks.
#  Tests different portfolio sizes in the first approximately 3 years
# then evaluates the performance on the remaining data.

# load packages
using CSV
using DataFrames
using JuMP
using GLPK
using Ipopt
using Statistics
using Plots

# Load and clean stock data from CSV
function load_stock_data(filepath)
    df = CSV.read(filepath, DataFrame)
    
    stocks = unique(df.Name)
    dates = sort(unique(df.date))
    
    println("Found ", length(stocks), " stocks and ", length(dates), " trading days")
    
    # Create close price matrix: rows=dates, columns=stocks
    price_matrix = Matrix{Float64}(undef, length(dates), length(stocks))
    
    for (stock_idx, stock) in enumerate(stocks)
        stock_data = filter(row -> row.Name == stock, df)
        stock_data = sort(stock_data, :date)
        
        for (date_idx, date) in enumerate(dates)
            matching_row = filter(row -> row.date == date, stock_data)
            price_matrix[date_idx, stock_idx] = isempty(matching_row) ? NaN : matching_row[1, :close]
        end
    end
    
    # Get rid of columns (stocks) with any NaN values
    valid_stocks_mask = [!any(isnan.(price_matrix[:, j])) for j in 1:size(price_matrix, 2)]
    
    # Filtered stocks and price matrix
    price_matrix = price_matrix[:, valid_stocks_mask]
    stocks = stocks[valid_stocks_mask]
    
    println("After cleaning: ", length(stocks), " stocks and ", length(dates), " trading days")
    
    return price_matrix, stocks, dates
end

# Calculate daily returns from price matrix
function calc_returns(prices)
    n_dates, n_stocks = size(prices)
    returns = Matrix{Float64}(undef, n_dates-1, n_stocks)
    
    for j in 1:n_stocks
        for i in 1:(n_dates-1)
            returns[i, j] = (prices[i+1, j] - prices[i, j]) / prices[i, j]
        end
    end
    
    return returns
end

# Select the top performing stocks by mean return
function select_top_stocks(returns, stocks, n_top)
    mean_returns = vec(mean(returns, dims=1))
    top_idx = sortperm(mean_returns, rev=true)[1:min(n_top, length(mean_returns))]
    
    return returns[:, top_idx], stocks[top_idx], mean_returns[top_idx]
end

# MAD portfolio optimization
function optimize_mad(returns, mean_returns)
    n_assets = size(returns, 2)
    n_periods = size(returns, 1)
    
    model = Model(GLPK.Optimizer)
    set_silent(model)
    
    # Decision variables
    @variable(model, w[1:n_assets] >= 0)     # portfolio weights (non-negative)
    @variable(model, u[1:n_periods] >= 0)    # positive deviations
    @variable(model, v[1:n_periods] >= 0)    # negative deviations
    
    # Minimize mean absolute deviation
    @objective(model, Min, sum(u[t] + v[t] for t in 1:n_periods) / n_periods)
    
    # Portfolio return deviations
    @constraint(model, deviations[t in 1:n_periods],
        sum(returns[t, i] * w[i] for i in 1:n_assets) == 
        sum(w[i] * mean_returns[i] for i in 1:n_assets) + u[t] - v[t])
    
    # Portfolio constraints
    @constraint(model, sum(w) == 1)          # weights sum to 1
    
    optimize!(model)
    
    return termination_status(model) == MOI.OPTIMAL ? [value(w[i]) for i in 1:n_assets] : nothing
end

# Mean Variance (Markowitz) optimization
function optimize_markowitz(returns, mean_returns; target_return=nothing)
    n_assets = size(returns, 2)
    cov_matrix = cov(returns)
    
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    
    @variable(model, w[1:n_assets] >= 0)     # non-negative weights
    @objective(model, Min, w' * cov_matrix * w)     # minimize variance
    @constraint(model, sum(w) == 1)                 # weights sum to 1
    
    if target_return !== nothing
        @constraint(model, mean_returns' * w >= target_return)
    end
    
    optimize!(model)
    
    weights = termination_status(model) in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED] ? 
              [value(w[i]) for i in 1:n_assets] : nothing
    
    return weights, cov_matrix
end

# Evaluate portfolio performance on test data
function backtest_portfolio(weights, test_returns)
    portfolio_returns = test_returns * weights
    
    # Calculate performance metrics
    returns_plus_one = 1 .+ portfolio_returns
    total_return = prod(returns_plus_one) - 1
    annual_return = (1 + total_return)^(252/length(portfolio_returns)) - 1
    volatility = std(portfolio_returns) * sqrt(252)
    sharpe = annual_return / volatility
    
    # Calculate max drawdown
    cumulative = cumprod(returns_plus_one)
    running_max = accumulate(max, cumulative)
    drawdowns = (cumulative .- running_max) ./ running_max
    max_dd = minimum(drawdowns)
    
    return annual_return, volatility, sharpe, max_dd
end

# Create pie chart for portfolio weights
function create_portfolio_pie_chart(weights, stock_names, method_name, n_stocks)
    # Only show stocks with meaningful weights (>= 1%)
    min_weight = 0.01
    significant_indices = findall(w -> w >= min_weight, weights)
    
    # Prepare data for plotting
    plot_weights = weights[significant_indices]
    plot_labels = stock_names[significant_indices]
    
    # Group small weights into "Others"
    other_weight = sum(weights[weights .< min_weight])
    
    if other_weight > 0.001
        plot_weights = vcat(plot_weights, other_weight)
        plot_labels = vcat(plot_labels, "Others")
    end
    
    # Convert to percentages for display
    plot_percentages = plot_weights * 100
    
    # Create the pie chart
    pie_chart = pie(plot_labels, plot_percentages,
                   title = string(method_name, " Portfolio (Top ", string(n_stocks), " stocks)"),
                   legend=:outertopright,
                   size=(800, 600))
    
    return pie_chart
end

# Main analysis function with pie charts
function run_analysis(filepath; create_charts=true)
    println("=== Portfolio Optimization Analysis ===")
    println("Comparing MAD vs Mean Variance optimization")
    
    # Load data
    prices, all_stocks, dates = load_stock_data(filepath)
    returns = calc_returns(prices)
    
    # Split into training (60%) and testing (40%)
    n_train = Int(floor(size(returns, 1) * 0.6))
    train_returns = returns[1:n_train, :]
    test_returns = returns[(n_train+1):end, :]
    
    println("Training: ", n_train, " days, Testing: ", size(test_returns, 1), " days")
    
    # Test different portfolio sizes
    portfolio_sizes = [75, 150, 200]
    
    println("\nResults Summary:")
    println("Portfolio | Method    | Return  | Risk    | Sharpe  | Max DD ")
    println("-" ^ 58)

    for n_stocks in portfolio_sizes
        # Select top performers
        selected_returns, selected_stocks, mean_returns = select_top_stocks(
            train_returns, all_stocks, n_stocks)
        
        # Get corresponding test returns for selected stocks
        stock_indices = [findfirst(x -> x == stock, all_stocks) for stock in selected_stocks]
        test_selected = test_returns[:, stock_indices]
        
        # MAD optimization
        mad_weights = optimize_mad(selected_returns, mean_returns)
        if mad_weights !== nothing
            ret, vol, sharpe, dd = backtest_portfolio(mad_weights, test_selected)
            @printf("Top %-4d | %-8s | %6.2f%% | %6.2f%% | %6.3f | %6.2f%%\n", 
                    n_stocks, "MAD", ret*100, vol*100, sharpe, dd*100)
            
            # Create pie chart for MAD
            if create_charts
                mad_chart = create_portfolio_pie_chart(mad_weights, selected_stocks, "MAD", n_stocks)
                display(mad_chart)
            end
        end
        
        # Mean Variance optimization  
        mv_weights, cov_mat = optimize_markowitz(selected_returns, mean_returns)
        if mv_weights !== nothing
            ret, vol, sharpe, dd = backtest_portfolio(mv_weights, test_selected)
            @printf("Top %-4d | %-8s | %6.2f%% | %6.2f%% | %6.3f | %6.2f%%\n", 
                    n_stocks, "Markowitz", ret*100, vol*100, sharpe, dd*100)
            
            # Create pie chart for Markowitz
            if create_charts
                mv_chart = create_portfolio_pie_chart(mv_weights, selected_stocks, "Mean-Variance", n_stocks)
                display(mv_chart)
            end
        end
        
        println()
    end
end

# Run analysis
println("Ready to analyze portfolio optimization strategies")
println("Usage: run_analysis(\"your_stock_data.csv\") - with pie charts")
println("Usage: run_analysis(\"your_stock_data.csv\", create_charts=false) - without charts")
println("This compares MAD and Mean Variance optimization on different portfolio sizes")