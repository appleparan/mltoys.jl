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
end

@testset "Date Validation" begin
    @testset "Invalid date range" begin
        # one month
        _year = 2018
        _month = 1
        from_date = ZonedDateTime(_year, _month, 1, 0, tz"Asia/Seoul")
        to_date = ZonedDateTime(_year, _month, Dates.daysinmonth(DateTime(_year, _month)), 23, tz"Asia/Seoul")
        dates = collect(from_date:Dates.Hour(1):to_date)
        df = DataFrame(date = dates)
        
        @test_throws ArgumentError validate_dates(to_date, from_date, 1, df)
    end

    @testset "Invalid window size" begin
        # one month
        _year = 2018
        _month = 1
        from_date = ZonedDateTime(_year, _month, 1, 0, tz"Asia/Seoul")
        to_date = ZonedDateTime(_year, _month, 1, 23, tz"Asia/Seoul")
        dates = collect(from_date:Dates.Hour(1):to_date)
        df = DataFrame(date = dates)
        
        @test_throws BoundsError validate_dates(from_date, to_date, 48, df)
    end
end

@testset "Window" begin
    @testset "Basic window" begin
        df = DataFrame(
            A = 1:12,
            B = 13:24,
            C = 25:36)
        sample_size = 4
        output_size = 2
        dfs = window_df(df, sample_size, output_size)
        @test length(dfs) == 7
        @test dfs[1] == DataFrame(
            A = 1:6,
            B = 13:18,
            C = 25:30)
        @test dfs[7] == DataFrame(
            A = 7:12,
            B = 19:24,
            C = 31:36)
    end

    @testset "Window + Date Range" begin
        # one month
        _year = 2018
        _month = 1
        from_date = ZonedDateTime(_year, _month, 1, 0, tz"Asia/Seoul")
        to_date = ZonedDateTime(_year, _month, Dates.daysinmonth(DateTime(_year, _month)), 23, tz"Asia/Seoul")
        dates = collect(from_date:Dates.Hour(1):to_date)
        n = length(dates)

        df = DataFrame(
            A = 1:n,
            B = n+1:2n,
            C = 2n+1:3n,
            date = dates
        )
        sample_size = 12
        output_size = 2
        dfs = window_df(df, sample_size, output_size,
            ZonedDateTime(_year, _month, 10, 0, 1, tz"Asia/Seoul"),
            ZonedDateTime(_year, _month, 14, 23, 59, tz"Asia/Seoul"))

        @test length(dfs) == 106
        @test dfs[1] == DataFrame(
            A = 218:231,
            B = 962:975,
            C = 1706:1719,
            date = collect(ZonedDateTime(_year, _month, 10, 1, tz"Asia/Seoul"):Hour(1):ZonedDateTime(_year, _month, 10, 14, tz"Asia/Seoul"))
        )
        @test dfs[end] == DataFrame(
            A = 323:336,
            B = 1067:1080,
            C = 1811:1824,
            date = collect(ZonedDateTime(_year, _month, 14, 10, tz"Asia/Seoul"):Hour(1):ZonedDateTime(_year, _month, 14, 23, tz"Asia/Seoul"))
        )
    end

    @testset "Window + Date Range (Last day)" begin
        # one month
        _year = 2018
        _month = 1
        from_date = ZonedDateTime(_year, _month, 1, 0, tz"Asia/Seoul")
        to_date = ZonedDateTime(_year, _month, Dates.daysinmonth(DateTime(_year, _month)), 23, tz"Asia/Seoul")
        dates = collect(from_date:Dates.Hour(1):to_date)
        n = length(dates)

        df = DataFrame(
            A = 1:n,
            B = n+1:2n,
            C = 2n+1:3n,
            date = dates
        )

        sample_size = 12
        output_size = 2
        window_size = sample_size + output_size
        dfs = window_df(df, sample_size, output_size,
            ZonedDateTime(_year, _month, 30, 0, tz"Asia/Seoul"),
            ZonedDateTime(_year, _month, 31, 23, tz"Asia/Seoul"))

        @test length(dfs) == 35
        @test dfs[35] == DataFrame(
            A = 731:744,
            B = 1475:1488,
            C = 2219:2232,
            date = collect(ZonedDateTime(_year, _month, 31, 10, tz"Asia/Seoul"):Hour(1):ZonedDateTime(_year, _month, 31, 23, tz"Asia/Seoul"))
        )
    end

    @testset "Window + Date Range + StationCode" begin
        # one month
        _year = 2018
        _month = 1
        from_date = ZonedDateTime(_year, _month, 1, 0, tz"Asia/Seoul")
        to_date = ZonedDateTime(_year, _month, Dates.daysinmonth(DateTime(_year, _month)), 23, tz"Asia/Seoul")
        dates = collect(from_date:Dates.Hour(1):to_date)
        n = length(dates)
        baseCode = 1000
        tot_stn = 10

        listA = 1:n
        listB = n+1:2n
        listC = 2n+1:3n

        # stationCode 1 1 1 1 2 2 2 2 3 3 3 3 ...
        # A, B, C, date 1 2 3 4 1 2 3 4 1 2 3 4 ...
        df = DataFrame(
            stationCode = repeat(range(baseCode, step=1, length=tot_stn), inner=n),
            A = repeat(listA, outer=tot_stn),
            B = repeat(listB, outer=tot_stn),
            C = repeat(listC, outer=tot_stn),
            date = repeat(dates, outer=tot_stn))
        
        sample_size = 12
        output_size = 2
        window_size = sample_size + output_size

        target_stncode = baseCode + abs(rand(Int)) % tot_stn
        dfs = window_df(df, sample_size, output_size,
            ZonedDateTime(_year, _month, 30, 0, tz"Asia/Seoul"),
            ZonedDateTime(_year, _month, 31, 23, tz"Asia/Seoul"),
            target_stncode)

        df = dfs[1]
        @test DataFrames.nrow(df) == window_size
        @test df[!, :stationCode] == ones(window_size) * target_stncode
    end
end

@testset "Splitting Sizes" begin
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
end

@testset "Input" begin
    @testset "getX & getY" begin
        sample_size = 4
        output_size = 3
        window_size = sample_size + output_size

        start = 10
        lists = collect(Iterators.partition((start+1):(start + 3*window_size), window_size))

        df = DataFrame(
            A = lists[1],
            B = lists[2],
            C = lists[3])
        features = [:A, :B]

        X = getX(df, features, sample_size)
        @test X == Matrix(hcat([11, 12, 13, 14], [18, 19, 20, 21]))

        Y = getY(df, :C, sample_size)
        @test Y == Vector([29, 30, 31])
    end
end

@testset "DNN Input" begin
    @testset "DNN Pair construction" begin
        sample_size = 24
        output_size = 12

        _year = 2015
        _month = 1
        from_date = ZonedDateTime(_year, _month, 1, 0, tz"Asia/Seoul")
        to_date = ZonedDateTime(_year, _month, Dates.daysinmonth(DateTime(_year, _month)), 23, tz"Asia/Seoul")
        dates = from_date:Dates.Hour(1):to_date
        len_dates = length(dates)
        lists = collect(Iterators.partition(1:3*len_dates, len_dates))
        eltype = Integer

        df = DataFrame(
            date = dates,
            A = lists[1],
            B = lists[2],
            C = lists[3])

        features = [:A, :B]
        ycol = :C

        wd = window_df(df, sample_size, output_size, from_date, to_date)
        # pair = ([1,...,24,len_df,...,len_df+24],[2*len_df,...,2*len_df+12])
        pair = make_pair_DNN(wd[1], ycol, features, sample_size, output_size, eltype)

        # X length
        @test length(pair[1]) == sample_size * length(features)
        # Y length
        @test length(pair[2]) == output_size

        @test pair[1] == reduce(vcat, [collect(1:sample_size), collect(len_dates + 1:len_dates + sample_size)])
        @test pair[2] == collect((sample_size + 2*len_dates + 1):(sample_size + 2*len_dates + output_size))
    end

    @testset "DNN Batch(size 2) construction" begin
        sample_size = 24
        output_size = 12
        batch_size = 2

        _year = 2015
        _month = 1
        from_date = ZonedDateTime(_year, _month, 1, 0, tz"Asia/Seoul")
        to_date = ZonedDateTime(_year, _month, Dates.daysinmonth(DateTime(_year, _month)), 23, tz"Asia/Seoul")
        dates = from_date:Dates.Hour(1):to_date
        len_dates = length(dates)
        lists = collect(Iterators.partition(1:3*len_dates, len_dates))
        eltype = Integer

        df = DataFrame(
            date = dates,
            A = lists[1],
            B = lists[2],
            C = lists[3])

        features = [:A, :B]
        ycol = :C

        # type contraint to Floating point
        for col in [:A, :B, :C]
            df[!, col] = Float32.(df[!, col])
        end

        wd = window_df(df, sample_size, output_size, from_date, to_date)

        batch = make_batch_DNN(wd[1:batch_size], :C, [:A, :B], sample_size, output_size, batch_size,
            0.5, eltype)
        
        # indices for comparison
        in_chnks = [1:sample_size,
                    2:(sample_size+1)]
        out_chnks = [(sample_size+1):(sample_size+output_size),
                     (sample_size+2):(sample_size+output_size+1)]

        # Test X
        @test batch[1] == hcat(
            vcat(df[in_chnks[1], :A], df[in_chnks[1], :B]),
            vcat(df[in_chnks[2], :A], df[in_chnks[2], :B]))
        # Test Y
        @test batch[2] == hcat(
            vcat(df[out_chnks[1], :C]),
            vcat(df[out_chnks[2], :C]))

        @test size(batch[1]) == (sample_size * 2, batch_size)
        @test size(batch[2]) == (output_size, batch_size)
    end

    @testset "DNN Batch(size 3) construction" begin
        sample_size = 24
        output_size = 12
        batch_size = 3

        _year = 2015
        _month = 1
        from_date = ZonedDateTime(_year, _month, 1, 0, tz"Asia/Seoul")
        to_date = ZonedDateTime(_year, _month, Dates.daysinmonth(DateTime(_year, _month)), 23, tz"Asia/Seoul")
        dates = from_date:Dates.Hour(1):to_date
        len_dates = length(dates)
        lists = collect(Iterators.partition(1:3*len_dates, len_dates))
        eltype = Integer

        df = DataFrame(
            date = dates,
            A = lists[1],
            B = lists[2],
            C = lists[3])

        features = [:A, :B]
        ycol = :C

        # type contraint to Floating point
        for col in [:A, :B, :C]
            df[!, col] = Float32.(df[!, col])
        end

        wd = window_df(df, sample_size, output_size, from_date, to_date)

        batch = make_batch_DNN(wd[1:batch_size], :C, [:A, :B], sample_size, output_size, batch_size,
            0.5, eltype)
        
        # indices for comparison
        in_chnks = [1:sample_size,
                    2:(sample_size+1),
                    3:(sample_size+2)]
        out_chnks = [(sample_size+1):(sample_size+output_size),
                     (sample_size+2):(sample_size+output_size+1),
                     (sample_size+3):(sample_size+output_size+2)]
        # Test X
        @test batch[1] == hcat(
            vcat(df[in_chnks[1], :A], df[in_chnks[1], :B]), 
            vcat(df[in_chnks[2], :A], df[in_chnks[2], :B]), 
            vcat(df[in_chnks[3], :A], df[in_chnks[3], :B]))
        @test batch[2] == hcat(
            vcat(df[out_chnks[1], :C]),
            vcat(df[out_chnks[2], :C]),
            vcat(df[out_chnks[3], :C]))
        @test size(batch[1]) == (sample_size * 2, 3)
        @test size(batch[2]) == (output_size, 3)
    end
end

@testset "Validate Pairs" begin
    @testset "validate_pairs_normal" begin
        X = Matrix([ 1  2  3  4;
                     7  8  9 10;
                    13 14 15 16;
                    19 20 21 22;
                    25 26 27 28])
        Y = Matrix([5  6  7  8  9 10])
        
        remove_sparse_input!(X, Y, 0.5)

        @test X == Matrix([ 1  2  3  4;
                     7  8  9 10;
                    13 14 15 16;
                    19 20 21 22;
                    25 26 27 28])
        @test Y == Matrix([5  6  7  8  9 10])
    end

    @testset "validate_pairs_missings" begin
        X = Matrix([ 1  2  3  4;
                     7  8  9 10;
                    13 14 15 16;
                    19 20 21 22;
                    25 26 27 28])
        Y = Matrix([23 24 25 missing missing missing])
        
        remove_sparse_input!(X, Y, 0.5)

        @test X == Matrix([ 0.0 0.0 0.0 0.0;
                            0.0 0.0 0.0 0.0;
                            0.0 0.0 0.0 0.0;
                            0.0 0.0 0.0 0.0;
                            0.0 0.0 0.0 0.0;])
        @test Y == Matrix([0.0 0.0 0.0 0.0 0.0 0.0])
    end
    
    @testset "validate_pairs_missings_0.3" begin
        X = Matrix([ 1  2  3  4;
                     7  8  9 10;
                    13 14 15 16;
                    19 20 21 22;
                    25 26 27 28])
        Y = Matrix([23 24 25 26 missing missing])
        
        remove_sparse_input!(X, Y, 0.3)

        @test X == Matrix([ 0.0 0.0 0.0 0.0;
                            0.0 0.0 0.0 0.0;
                            0.0 0.0 0.0 0.0;
                            0.0 0.0 0.0 0.0;
                            0.0 0.0 0.0 0.0;])
        @test Y == Matrix([0.0 0.0 0.0 0.0 0.0 0.0])
    end

    @testset "validate_pairs_mx" begin
        X = Matrix([ 1  2  3  4;
                     7  8  9 10;
                    13 14 15 16;
                    19 20 21 22;
                    25 26 27 28])
        Y = Matrix([23 24 25 0.0 missing missing])
        
        remove_sparse_input!(X, Y, 0.5)

        @test X == Matrix([ 0.0 0.0 0.0 0.0;
                            0.0 0.0 0.0 0.0;
                            0.0 0.0 0.0 0.0;
                            0.0 0.0 0.0 0.0;
                            0.0 0.0 0.0 0.0;])
        @test Y == Matrix([0.0 0.0 0.0 0.0 0.0 0.0])
    end
end

@testset "CNN to RNN Batch Input Conversion" begin
    @testset "Batch Serialization" begin
        m = 3
        n = 5
        a = reshape(1:m*n, m, n)
        @test size(a) == (m, n)

        # 2D -> Arrays of 1D
        b = serializeBatch(a)
        @test size(b) == (n,)
        @test size(collect(b[1])) == (m,)
    end

    @testset "Batch Serialization" begin
        m1 = 3
        m2 = 4
        n = 5
        X = reshape(1:m1*n, m1, n)
        Y = reshape(101:(100+m2*n), m2, n)
        Z = serializeBatch(X, Y)
        # Z => 
        @show Z
        @test size(Z) == (n,)
        @test size(collect(Z[1])) == (2,)
        # row size should be maintained
        @test size(Z[1][1]) == (m1,)
        @test size(Z[1][2]) == (m2,)
    end
end
