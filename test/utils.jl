using Base.Iterators: partition
using Test

using DataFrames, Query
using StatsBase: zscore, mean, std, mean_and_std
using Dates, TimeZones

using MLToys

@info "Testing Stats..."
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
        MLToys.zscore!(df, "A", "newA")
        @test df[:newA] == zscore(a)
    end
end

#=
@info "Testing hampel..."
@testset "hampel" begin
    @testset "hampel_single" begin
        df = DataFrame(
            A = 1:4,
            B = 5:8,
            C = 9:12)
        hampel!(df, "A", "nA")
        # @test isapprox(df[:nA], [0.494191, 0.498064, 0.501936, 0.505809])
    end
end
=#
@info "Testing zscore..."
@testset "zscore" begin
    @testset "zscore_single_string" begin
        df = DataFrame(
            A = 1:4,
            B = 5:8,
            C = 9:12)
        zscore!(df, "A", "nA")
        @test isapprox(df[:nA], [ -1.161895003862225,
            -0.3872983346207417,
             0.3872983346207417,
             1.161895003862225])
    end

    @testset "zscore_single_symbol" begin
        df = DataFrame(
            A = 1:4,
            B = 5:8,
            C = 9:12)
        zscore!(df, :A, "nA")
        @test isapprox(df[:nA], [ -1.161895003862225,
            -0.3872983346207417,
             0.3872983346207417,
             1.161895003862225])
    end

    @testset "zscore_Array_string" begin
        df = DataFrame(
            A = 1:4,
            B = 5:8,
            C = 9:12)
        zscore!(df, ["A", "B"], ["nA", "nB"])
        @test isapprox(df[:nA], [ -1.161895003862225,
            -0.3872983346207417,
             0.3872983346207417,
             1.161895003862225])
        @test isapprox(df[:nB], [ -1.161895003862225,
             -0.3872983346207417,
              0.3872983346207417,
              1.161895003862225])
    end

    @testset "zscore_Array_symbol" begin
        df = DataFrame(
            A = 1:4,
            B = 5:8,
            C = 9:12)
        zscore!(df, [:A, :B], [:nA, :nB])
        @test isapprox(df[:nA], [ -1.161895003862225,
            -0.3872983346207417,
             0.3872983346207417,
             1.161895003862225])
        @test isapprox(df[:nB], [ -1.161895003862225,
             -0.3872983346207417,
              0.3872983346207417,
              1.161895003862225])
    end
end

@testset "Split Sets" begin
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

    @testset "window_df" begin
        df = DataFrame(
            A = 1:12,
            B = 13:24,
            C = 25:36)
        sample_size = 4
        idxs = window_df(df, sample_size)
        @test length(idxs) == 9
        @test df[collect(idxs[1]), :] == DataFrame(
            A = 1:4,
            B = 13:16,
            C = 25:28)
        @test df[collect(idxs[4]), :] == DataFrame(
            A = 4:7,
            B = 16:19,
            C = 28:31)
        @test df[collect(idxs[9]), :] == DataFrame(
            A = 9:12,
            B = 21:24,
            C = 33:36)
    end

    @testset "window_df_w_dates" begin
        # one month
        year = 2018
        month = 1
        n = Dates.daysinmonth(DateTime(year, month)) * 24
        df = DataFrame(
            A = 1:n,
            B = n+1:2n,
            C = 2n+1:3n,
            date = 
                collect(ZonedDateTime(year, month, 1, 0,
                    tz"Asia/Seoul"):Hour(1):ZonedDateTime(year, month, 1, 0, tz"Asia/Seoul") + Hour(n - 1))
        )
        sample_size = 12
        idxs = window_df(df, sample_size, ZonedDateTime(year, month, 10, 0, tz"Asia/Seoul"), ZonedDateTime(year, month, 14, 23, tz"Asia/Seoul"))
        @test length(idxs) == 109
        @test df[collect(idxs[1]), :] == DataFrame(
            A = 217:228,
            B = 961:972,
            C = 1705:1716,
            date = collect(ZonedDateTime(year, month, 10, 0, tz"Asia/Seoul"):Hour(1):ZonedDateTime(year, month, 10, 11, tz"Asia/Seoul"))
        )
        @test df[collect(idxs[109]), :] == DataFrame(
            A = 325:336,
            B = 1069:1080,
            C = 1813:1824,
            date = collect(ZonedDateTime(year, month, 14, 12, tz"Asia/Seoul"):Hour(1):ZonedDateTime(year, month, 14, 23, tz"Asia/Seoul"))
        )
    end

    @testset "window_df_w_dates_lastday" begin
        # one month
        year = 2018
        month = 1
        n = Dates.daysinmonth(DateTime(year, month)) * 24
        df = DataFrame(
            A = 1:n,
            B = n+1:2n,
            C = 2n+1:3n,
            date = 
                collect(ZonedDateTime(year, month, 1, 0,
                    tz"Asia/Seoul"):Hour(1):ZonedDateTime(year, month, 1, 0, tz"Asia/Seoul") + Hour(n - 1))
        )
        sample_size = 12
        idxs = window_df(df, sample_size,
            ZonedDateTime(year, month, 30, 0, tz"Asia/Seoul"), ZonedDateTime(year, month, 31, 23, tz"Asia/Seoul"))
        @test length(idxs) == 37
        @test df[collect(idxs[37]), :] == DataFrame(
            A = 733:744,
            B = 1477:1488,
            C = 2221:2232,
            date = collect(ZonedDateTime(year, month, 31, 12, tz"Asia/Seoul"):Hour(1):ZonedDateTime(year, month, 31, 23, tz"Asia/Seoul"))
        )
    end

    @testset "window_df_w_dates_exception_01" begin
        year = 2018
        month = 1
        n = Dates.daysinmonth(DateTime(year, month)) * 24
        df = DataFrame(
            A = 1:n,
            B = n+1:2n,
            C = 2n+1:3n
        )
        @test_throws UndefVarError window_df(df, sample_size,
            ZonedDateTime(year, month, 31, 1, tz"Asia/Seoul"), ZonedDateTime(year, month, 31, 23, tz"Asia/Seoul"))
    end

    @testset "window_df_w_dates_exception_02" begin
        # one month
        year = 2018
        month = 1
        n = Dates.daysinmonth(DateTime(year, month)) * 24
        df = DataFrame(
            A = 1:n,
            B = n+1:2n,
            C = 2n+1:3n,
            date = 
                collect(ZonedDateTime(year, month, 1, 0,
                    tz"Asia/Seoul"):Hour(1):ZonedDateTime(year, month, 1, 0, tz"Asia/Seoul") + Hour(n - 1))
        )
        sample_size = 12
        @test_throws ArgumentError window_df(df, sample_size,
            ZonedDateTime(year, month, 31, 23, tz"Asia/Seoul"), ZonedDateTime(year, month, 31, 1, tz"Asia/Seoul"))
        sample_size = 36
        @test_throws BoundsError window_df(df, sample_size,
            ZonedDateTime(year, month, 31, 1, tz"Asia/Seoul"), ZonedDateTime(year, month, 31, 23, tz"Asia/Seoul"))
        
    end

    @testset "split_sizes3_mod0" begin
        total_size = 100
        batch_size = 10
        train_size, valid_size, test_size = split_sizes3(total_size, batch_size)
        @test train_size == 64
        @test valid_size == 16
        @test test_size == 20
    end

    @testset "split_sizes3_prime" begin
        total_size = 97
        batch_size = 10
        train_size, valid_size, test_size = split_sizes3(total_size, batch_size)
        @test train_size == 62
        @test valid_size == 16
        @test test_size == 19
    end

    @testset "split_sizes2_mod0" begin
        total_size = 100
        batch_size = 10
        train_size, valid_size = split_sizes2(total_size, batch_size)
        @test train_size == 80
        @test valid_size == 20
    end

    @testset "split_sizes2_prime" begin
        total_size = 97
        batch_size = 10
        train_size, valid_size = split_sizes2(total_size, batch_size)
        @test train_size == 78
        @test valid_size == 19
    end

    @testset "create_idxs3" begin
        total_size = 97
        batch_size = 10
        # for test, don't permute
        total_idx = collect(1:total_size)
        train_size, valid_size, test_size = split_sizes3(total_size, batch_size)
        train_idxs, valid_idxs, test_idxs = create_idxs(total_idx, train_size, valid_size, test_size)

        @test train_idxs == collect(1:62)
        @test valid_idxs == collect(63:78)
        @test test_idxs == collect(79:97)
    end

    @testset "create_idxs2" begin
        total_size = 97
        batch_size = 10
        # for test, don't permute
        total_idx = collect(1:total_size)
        train_size, valid_size = split_sizes2(total_size, batch_size)
        train_idxs, valid_idxs = create_idxs(total_idx, train_size, valid_size)

        @test train_idxs == collect(1:78)
        @test valid_idxs == collect(79:97)
    end

    @testset "create_chunks3" begin
        total_size = 97
        batch_size = 10
        # for test, don't permute
        total_idx = collect(1:total_size)
        train_size, valid_size, test_size = split_sizes3(total_size, batch_size)
        train_chnks, valid_chnks, test_chnks = create_chunks(total_idx, train_size, valid_size, test_size, batch_size)

        @test length(train_chnks) == 7
        @test length(valid_chnks) == 2
        @test length(test_chnks) == 2

        @test train_chnks[1] == collect(1:10)
        @test train_chnks[7] == collect(61:62)
        @test valid_chnks[1] == collect(63:72)
        @test valid_chnks[2] == collect(73:78)
        @test test_chnks[1] == collect(79:88)
        @test test_chnks[2] == collect(89:97)
    end
    
    @testset "create_chunks2" begin
        total_size = 97
        batch_size = 10
        # for test, don't permute
        total_idx = collect(1:total_size)
        train_size, valid_size = split_sizes2(total_size, batch_size)
        train_chnks, valid_chnks = create_chunks(total_idx, train_size, valid_size, batch_size)

        @test length(train_chnks) == 8
        @test length(valid_chnks) == 2

        @test train_chnks[1] == collect(1:10)
        @test train_chnks[8] == collect(71:78)
        @test valid_chnks[1] == collect(79:88)
        @test valid_chnks[2] == collect(89:97)
    end
end

@testset "Pairs" begin
    @testset "getX" begin
        df = DataFrame(
            A = 1:12,
            B = 13:24,
            C = 25:36)
        sample_size = 4
        idxs = collect(1:4)
        features = ["A", "B", "C"]
        X = getX(df, idxs, features)
        @test X == [1, 2, 3, 4, 13, 14, 15, 16, 25, 26, 27, 28]
        idxs = collect(3:6)
        X = getX(df, idxs, features)
        @test X == [3, 4, 5, 6, 15, 16, 17, 18, 27, 28, 29, 30]
    end

    @testset "getHoursLater_1" begin
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
    
    @testset "getHoursLater_2" begin
        date_fmt = Dates.DateFormat("yyyy-mm-dd HH:MM:SSz")
        df = DataFrame(
            date = ZonedDateTime.([
                "2015-01-01 01:00:00+09:00",
                "2015-01-01 02:00:00+09:00",
                "2015-01-01 03:00:00+09:00",
                "2015-01-01 04:00:00+09:00",
                "2015-01-01 05:00:00+09:00",
                "2015-01-01 06:00:00+09:00",
                "2015-01-01 07:00:00+09:00",
                "2015-01-01 08:00:00+09:00",
                "2015-01-01 09:00:00+09:00",
                "2015-01-01 10:00:00+09:00",
                "2015-01-01 11:00:00+09:00",
                "2015-01-01 12:00:00+09:00"], date_fmt)
        )
        hours = 3
        last_date_str = "2015-01-01 04:00:00+09:00"
        df_hours = getHoursLater(df, hours, last_date_str, date_fmt)
        @test size(df_hours)[1] == hours
        @test df_hours == DataFrame(
            date = ZonedDateTime.([
            "2015-01-01 05:00:00+09:00",
            "2015-01-01 06:00:00+09:00",
            "2015-01-01 07:00:00+09:00"], date_fmt)
        )
    end
end

@testset "Minibatch" begin
    @testset "make_pair" begin
        sample_size = 24
        input_size = sample_size * 2
        output_size = 12

        date_fmt = Dates.DateFormat("yyyy-mm-ddTHH:MM:SSz")
        Jan_2015 = ZonedDateTime("2015-01-01T00:00:00+09:00", date_fmt)
        Jan_2015_hours = collect(Jan_2015:Hour(1):Jan_2015 + Day(30))
        len_df = length(Jan_2015_hours)

        df = DataFrame(
            date = Jan_2015_hours,
            A = collect(         1:  len_df),
            B = collect(  len_df+1:2*len_df),
            C = collect(2*len_df+1:3*len_df)
        )
        idx = collect(1:sample_size)
        pair = make_pairs(df, :C, idx, [:A, :B], input_size, output_size)
        #=
        pair should be..

        ([1,...,24,len_df,...,len_df+24],[2*len_df,...,2*len_df+12])
        =#
        @test length(pair[1]) == input_size
        @test length(pair[2]) == output_size
        @test pair[1] == reduce(vcat, [collect(1:sample_size), collect(len_df + 1:len_df + sample_size)])
        @test pair[2] == collect((sample_size + 2*len_df + 1):(sample_size + 2*len_df + output_size))
    end

    @testset "minibatch_list" begin
        pairs = [
            ([ 1, 2, 3, 4], [ 5, 6]),
            ([ 7, 8, 9,10], [11,12]),
            ([13,14,15,16], [17,18]),
            ([19,20,21,22], [23,24]),
            ([25,26,27,28], [29,30])]

        minibatch = make_minibatch(pairs, [1,2], 2)
        @test minibatch == 
            ([1 7; 2 8; 3 9; 4 10], [5 11; 6 12])
        minibatch = make_minibatch(pairs, [3,4], 2)
        @test minibatch == 
            ([13 19; 14 20; 15 21; 16 22], [17 23; 18 24])
        minibatch = make_minibatch(pairs, [5], 2)
        @test minibatch == 
            ([25 0; 26 0; 27 0; 28 0], [29 0; 30 0])

        minibatch = make_minibatch(pairs, [1,2,4], 3)
        @test minibatch == 
            ([1 7 19; 2 8 20; 3 9 21; 4 10 22], [5 11 23; 6 12 24])
        minibatch = make_minibatch(pairs, [4,5], 3)
        @test minibatch == 
            ([19 25 0; 20 26 0; 21 27 0; 22 28 0], [23 29 0; 24 30 0])
    end
end
