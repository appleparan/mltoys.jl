using Base.Iterators: partition
using Test

using DataFrames, Query
using StatsBase: zscore, mean, std, mean_and_std

using Dates, TimeZones
using MLToys: standardize!, exclude_elem

@testset "Stats" begin
    @testset "zscore" begin
        a = collect(1:5)
        @test mean(a) == 3.0
        @test std(a) == sqrt(sum(abs2.(a .- mean(a)) / (length(a) - 1)))
    end 

    @testset "zscore_df" begin
        a = 1:5
        df = DataFrame()

        df[:A] = a 
        standardize!(df, "A", "newA")
        @test df[:newA] == zscore(a)
    end
end

@testset "Split" begin
    @testset "split_cols" begin
        cols = ["A", "B", "C", "D"]
        target = "A"
        exclude_target = exclude_elem(cols, target)
        @test cols == ["A", "B", "C", "D"]
        @test exclude_target == ["B", "C", "D"]
        
        target = "C"
        exclude_target2 = exclude_elem(cols, target)
        @test exclude_target2 == ["A", "B", "D"]
    end

    @testset "split_df" begin
        df = DataFrame(
            A = 1:12,
            B = 13:24,
            C = 25:36)
        sample_size = 4
        splitted, idxs = split_df(df, sample_size)
        @test splitted[1] == DataFrame(
            A = 1:4,
            B = 13:16,
            C = 25:28)
        @test splitted[3] == DataFrame(
            A = 9:12,
            B = 21:24,
            C = 33:36)
    end

    @testset "create_df_window" begin
        df = DataFrame(
            A = 1:12,
            B = 13:24,
            C = 25:36)
        sample_size = 4
        splitted, idxs = window_df(df, sample_size)
        @test length(idxs) == 4
        @test splitted[1] == DataFrame(
            A = 1:4,
            B = 13:16,
            C = 25:28)
        @test splitted[3] == DataFrame(
            A = 3:6,
            B = 15:18,
            C = 27:30)
        @test splitted[4] == DataFrame(
            A = 4:7,
            B = 16:19,
            C = 28:31)
    end
end

@testset "Minibatch" begin
    @testset "HoursLater" begin
        date_fmt = Dates.DateFormat("yyyy-mm-dd HH:MM:SSz")
        last_date_str = "2015-01-14 04:00:00+09:00"
        Jan_2015 = ZonedDateTime("2015-01-01 00:00:00+09:00", date_fmt)
        date_range = collect(Jan_2015:Hour(1):Jan_2015 + Day(30))
        df = DataFrame(date = date_range)

        df_hours = getHoursLater(df, 5, last_date_str)
        last_date = ZonedDateTime(last_date_str, date_fmt)

        @test df_hours == DataFrame(date = 
            [ZonedDateTime("2015-01-14 05:00:00+09:00", date_fmt),
             ZonedDateTime("2015-01-14 06:00:00+09:00", date_fmt),
             ZonedDateTime("2015-01-14 07:00:00+09:00", date_fmt),
             ZonedDateTime("2015-01-14 08:00:00+09:00", date_fmt),
             ZonedDateTime("2015-01-14 09:00:00+09:00", date_fmt)])
    end

    @testset "split" begin
        df = DataFrame(
            A = 1:12,
            B = 13:24,
            C = 25:36)
        cols = ["A", "B", "C"]
        target_col = "B"
        exec_cols = exclude_elem(cols, target_col)
        @test exec_cols == ["A", "C"]
        
        sample_size = 2
        batch_size = 3
        splitted, mb_idxs = split_df(df, sample_size)
    end
end
