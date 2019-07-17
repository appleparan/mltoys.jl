@testset "Stats" begin
    @testset "zscore" begin
        a = collect(1:5)
        @test mean(a) == 3.0
        @test std(a) == sqrt(sum(abs2.(a .- mean(a)) / (length(a) - 1)))
    end 

    @testset "zscore_df" begin
        a = 1:5
        df = DataFrame(A = a)
        zscore!(df, "A", "newA")
        @test df[:, :newA] == zscore(a)
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

@testset "zscore" begin
    @testset "zscore_single_string" begin
        df = DataFrame(
            A = 1:4,
            B = 5:8,
            C = 9:12)
        zscore!(df, "A", "nA")
        @test isapprox(df[:, :nA], [ -1.161895003862225,
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
        @test isapprox(df[:, :nA], [ -1.161895003862225,
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
        @test isapprox(df[:, :nA], [ -1.161895003862225,
            -0.3872983346207417,
             0.3872983346207417,
             1.161895003862225])
        @test isapprox(df[:, :nB], [ -1.161895003862225,
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
        @test isapprox(df[:, :nA], [ -1.161895003862225,
            -0.3872983346207417,
             0.3872983346207417,
             1.161895003862225])
        @test isapprox(df[:, :nB], [ -1.161895003862225,
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
        output_size = 2
        idxs = window_df(df, sample_size, output_size)
        @test length(idxs) == 7
        @test df[collect(idxs[1]), :] == DataFrame(
            A = 1:6,
            B = 13:18,
            C = 25:30)
        @test df[collect(idxs[7]), :] == DataFrame(
            A = 7:12,
            B = 19:24,
            C = 31:36)
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
        output_size = 2
        idxs = window_df(df, sample_size, output_size, ZonedDateTime(year, month, 10, 0, 1, tz"Asia/Seoul"), ZonedDateTime(year, month, 14, 23, 59, tz"Asia/Seoul"))
        @test length(idxs) == 106
        @test df[collect(idxs[1]), :] == DataFrame(
            A = 218:231,
            B = 962:975,
            C = 1706:1719,
            date = collect(ZonedDateTime(year, month, 10, 1, tz"Asia/Seoul"):Hour(1):ZonedDateTime(year, month, 10, 14, tz"Asia/Seoul"))
        )
        @test df[collect(idxs[106]), :] == DataFrame(
            A = 323:336,
            B = 1067:1080,
            C = 1811:1824,
            date = collect(ZonedDateTime(year, month, 14, 10, tz"Asia/Seoul"):Hour(1):ZonedDateTime(year, month, 14, 23, tz"Asia/Seoul"))
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

        @test df[end, :date] == ZonedDateTime(year, month, 31, 23, tz"Asia/Seoul")

        sample_size = 12
        output_size = 2
        idxs = window_df(df, sample_size, output_size,
            ZonedDateTime(year, month, 30, 0, tz"Asia/Seoul"), ZonedDateTime(year, month, 31, 23, tz"Asia/Seoul"))
        @test length(idxs) == 35
        @test df[collect(idxs[35]), :] == DataFrame(
            A = 731:744,
            B = 1475:1488,
            C = 2219:2232,
            date = collect(ZonedDateTime(year, month, 31, 10, tz"Asia/Seoul"):Hour(1):ZonedDateTime(year, month, 31, 23, tz"Asia/Seoul"))
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
        sample_size = 12
        output_size = 2
        @test_throws ArgumentError window_df(df, sample_size, output_size,
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
        output_size = 2
        @test_throws ArgumentError window_df(df, sample_size, output_size,
            ZonedDateTime(year, month, 31, 23, tz"Asia/Seoul"), ZonedDateTime(year, month, 31, 1, tz"Asia/Seoul"))
        sample_size = 36
        ouptut_size = 2
        @test_throws BoundsError window_df(df, sample_size, output_size,
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
        window_size = 12
        # for test, don't permute
        total_idxs = [i:(i+window_size-1) for i in 1:total_size]
        train_size, valid_size, test_size = split_sizes3(total_size, batch_size)
        train_idxs, valid_idxs, test_idxs = create_idxs(total_idxs, train_size, valid_size, test_size)

        # idxs::Array{<:UnitRange{Integer}, 1} : [1:2, 2:3, ...]
        @test train_idxs == [i:(i+window_size-1) for i in 1:train_size]
        @test valid_idxs == [i:(i+window_size-1) for i in (train_size+1):(train_size+valid_size)]
        @test test_idxs == [i:(i+window_size-1) for i in (train_size+valid_size+1):total_size]
    end

    @testset "create_idxs2" begin
        total_size = 97
        batch_size = 10
        window_size = 12
        # total_idxs = [1:12, 2:13, ..., 97:108]
        total_idxs = [i:(i+window_size-1) for i in 1:total_size]
        train_size, valid_size = split_sizes2(total_size, batch_size)
        train_idxs, valid_idxs = create_idxs(total_idxs, train_size, valid_size)

        @test train_idxs == [i:(i+window_size-1) for i in 1:train_size]
        @test valid_idxs == [i:(i+window_size-1) for i in (train_size+1):total_size]
    end

    @testset "create_idxs2_partial" begin
        total_size = 97
        batch_size = 10
        idx_begin = 11  
        idx_end = 30
        window_size = 12
        # total_idxs = [11:22, 12:23, ..., 30:41]
        total_idxs = [i:(i+window_size-1) for i in idx_begin:idx_end]
        total_size = length(total_idxs)

        train_size, valid_size = split_sizes2(total_size, batch_size)

        @test train_size == 16
        @test valid_size == 4

        train_idxs, valid_idxs = create_idxs(total_idxs, train_size, valid_size)

        @test train_idxs == [i:(i+window_size-1)
            for i in idx_begin:(idx_begin+train_size-1)]
        @test valid_idxs == [i:(i+window_size-1)
            for i in (idx_begin+train_size):(idx_begin+train_size+valid_size-1)]
    end
    
    @testset "create_chunks3" begin
        total_size = 97
        batch_size = 10
        window_size = 12
        # create windowed index
        total_idxs = [i:(i+window_size-1) for i in 1:total_size]
        # if total_size == 97 ->  train_size = 62, valid_size = 16, test_size = 19
        train_chnks, valid_chnks, test_chnks = create_chunks(total_idxs, 62, 16, 19, batch_size)

        @test length(train_chnks) == 7
        @test length(valid_chnks) == 2
        @test length(test_chnks) == 2

        # [1:12, 2:13, ..., 10:22] if batch_size = 10
        @test train_chnks[1] == [i:(i+window_size-1) for i in 1:10]
        # [61:72, 62:73, 63:74, 64:75] if batch_size = 10
        @test train_chnks[7] == [i:(i+window_size-1) for i in 61:62]
        # [65:76, 66:77, ..., 74:85] if batch_size = 10
        @test valid_chnks[1] == [i:(i+window_size-1) for i in 63:72]
        # [75:86, 76:87, ..., 80:91] if batch_size = 10
        @test valid_chnks[2] == [i:(i+window_size-1) for i in 73:78]
        # [81:92, 82:93, ..., 90:101] if batch_size = 10
        @test test_chnks[1] == [i:(i+window_size-1) for i in 79:88]
        # [91:102, 92:103, ..., 97:108] if batch_size = 10
        @test test_chnks[2] == [i:(i+window_size-1) for i in 89:97]
    end
    
    @testset "create_chunks2" begin
        total_size = 97
        batch_size = 10
        window_size = 12
        # create windowed index
        total_idxs = [i:(i+window_size-1) for i in 1:total_size]
        # if total_size == 97 ->  train_size =  78, valid_size = 19
        train_chnks, valid_chnks = create_chunks(total_idxs, 78, 19, batch_size)

        @test length(train_chnks) == 8
        @test length(valid_chnks) == 2

        # [1:12, 2:13, ..., 10:22] if batch_size = 10
        @test train_chnks[1] == [i:(i+window_size-1) for i in 1:10]
        # [71:82, 72:83, ..., 78:89] if batch_size = 10
        @test train_chnks[8] == [i:(i+window_size-1) for i in 71:78]
        # [79:90, 80:91, ..., 88:99] if batch_size = 10
        @test valid_chnks[1] == [i:(i+window_size-1) for i in 79:88]
        # [89:100, 90:101, ..., 97:108] if batch_size = 10
        @test valid_chnks[2] == [i:(i+window_size-1) for i in 89:97]
    end
end

@testset "Hours Manipulation" begin
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

@testset "Input" begin
    @testset "getX" begin
        df = DataFrame(
            A = 1:12,
            B = 13:24,
            C = 25:36)
        sample_size = 4
        idx = 1:4
        features = [:A, :B, :C]

        X = getX(df, idx, features, sample_size)
        @test X == Matrix(hcat([1, 2, 3, 4], [13, 14, 15, 16], [25, 26, 27, 28]))

        idx = 3:6
        X = getX(df, idx, features, sample_size)
        @test X == Matrix(hcat([3, 4, 5, 6], [15, 16, 17, 18], [27, 28, 29, 30]))
    end

    @testset "getY" begin
        len_row = 36
        test_dates = collect(ZonedDateTime(2015, 1, 1, tz"Asia/Seoul"):Hour(1): ZonedDateTime(2015, 1, 1, tz"Asia/Seoul")+Hour(35))
        df = DataFrame(
            date = test_dates,
            A = 1:36,
            B = 37:72,
            C = 73:108)

        sample_size = 8
        output_size = 4
        features = [:A, :B]
        ycol = :C

        idx = 1:sample_size
        Y = getY(df, idx, ycol, sample_size, output_size)
        @test Y == Array([81, 82, 83, 84])

        idx = 9:(9+sample_size)
        Y = getY(df, idx, ycol, sample_size, output_size)
        @test Y == Array([89, 90, 91, 92])
    end
end

@testset "DNN Input" begin
    @testset "DNN_pair" begin
        sample_size = 24
        output_size = 12

        Jan_2015 = ZonedDateTime(2015, 1, 1, tz"Asia/Seoul")
        Jan_2015_hours = collect(Jan_2015:Hour(1):(Jan_2015 + Day(30)))
        len_df = length(Jan_2015_hours)

        df = DataFrame(
            date = Jan_2015_hours,
            A = collect(         1:  len_df),
            B = collect(  len_df+1:2*len_df),
            C = collect(2*len_df+1:3*len_df))
        idx = 1:sample_size
        

        # pair = ([1,...,24,len_df,...,len_df+24],[2*len_df,...,2*len_df+12])
        pair = make_pair_DNN(df, :C, idx, [:A, :B], sample_size, output_size)

        # X length
        @test length(pair[1]) == sample_size * length([:A, :B])
        # Y length
        @test length(pair[2]) == output_size

        @test pair[1] == reduce(vcat, [collect(1:sample_size), collect(len_df + 1:len_df + sample_size)])
        @test pair[2] == collect((sample_size + 2*len_df + 1):(sample_size + 2*len_df + output_size))
    end

    @testset "DNN_batch_batch2" begin
        sample_size = 24
        output_size = 12
        batch_size = 2

        Jan_2015 = ZonedDateTime(2015, 1, 1, tz"Asia/Seoul")
        Jan_2015_hours = collect(Jan_2015:Hour(1):(Jan_2015 + Day(30)))
        len_df = length(Jan_2015_hours)

        df = DataFrame(
            date = Jan_2015_hours,
            A = collect(         1:  len_df),
            B = collect(  len_df+1:2*len_df),
            C = collect(2*len_df+1:3*len_df))

        # new syntax for setindex!
        for col in [:A, :B, :C]
            df[!, col] = Float32.(df[!, col])
        end
        idx = 1:sample_size
        # input indicies for test (X)
        chnks = [1:sample_size, 2:(sample_size+1)]
        # output indicies for test (Y)
        out_chnks = [(1+sample_size):(sample_size+output_size),
                     (2+sample_size):(sample_size+1+output_size)]

        batch = make_batch_DNN(df, :C, chnks, [:A, :B], sample_size, output_size, batch_size,
            0.5, Float64)

        # Test X
        @test batch[1] == hcat(
            vcat(df[chnks[1], :A], df[chnks[1], :B]),
            vcat(df[chnks[2], :A], df[chnks[2], :B]))
        @test batch[2] == hcat(
            vcat(df[out_chnks[1], :C]),
            vcat(df[out_chnks[2], :C]))
        @test size(batch[1]) == (sample_size * 2, 2)
        @test size(batch[2]) == (output_size, 2)
    end

    @testset "DNN_batch_batch3" begin
        sample_size = 24
        output_size = 12
        batch_size = 3

        Jan_2015 = ZonedDateTime(2015, 1, 1, tz"Asia/Seoul")
        Jan_2015_hours = collect(Jan_2015:Hour(1):(Jan_2015 + Day(30)))
        len_df = length(Jan_2015_hours)

        df = DataFrame(
            date = Jan_2015_hours,
            A = collect(         1:  len_df),
            B = collect(  len_df+1:2*len_df),
            C = collect(2*len_df+1:3*len_df))

        # new syntax for setindex!
        for col in [:A, :B, :C]
            df[!, col] = Float32.(df[!, col])
        end

        idx = 1:sample_size
        # input indicies for test (X)
        chnks = [1:sample_size, 2:(sample_size+1), 3:(sample_size+2)]
        # output indicies for test (Y)
        out_chnks = [(1+sample_size):(sample_size+output_size),
                     (2+sample_size):(sample_size+1+output_size),
                     (3+sample_size):(sample_size+2+output_size)]

        batch = make_batch_DNN(df, :C, chnks, [:A, :B], sample_size, output_size, batch_size,
            0.5, Float64)

        # Test X
        @test batch[1] == hcat(
            vcat(df[chnks[1], :A], df[chnks[1], :B]), 
            vcat(df[chnks[2], :A], df[chnks[2], :B]), 
            vcat(df[chnks[3], :A], df[chnks[3], :B]))
        @test batch[2] == hcat(
            vcat(df[out_chnks[1], :C]),
            vcat(df[out_chnks[2], :C]),
            vcat(df[out_chnks[3], :C]))
        @test size(batch[1]) == (sample_size * 2, 3)
        @test size(batch[2]) == (output_size, 3)
    end

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
#=
@testset "LSTM input" begin
    @testset "getX_LSTM" begin
        df = DataFrame(
            A = 1:12,
            B = 13:24,
            C = 25:36)
        sample_size = 4
        idx = 1:4
        features = [:A, :B, :C]
        X = getX_LSTM(df, idx, features, sample_size)
        @test X == [[1, 2, 3, 4] [13, 14, 15, 16] [25, 26, 27, 28]]

        idx = 3:6
        X = getX_LSTM(df, idx, features, sample_size)
        @test X == [[3, 4, 5, 6] [15, 16, 17, 18] [27, 28, 29, 30]]
    end

    @testset "make_input_LSTM" begin
        sample_size = 24
        input_size = sample_size * 2
        output_size = 12

        Jan_2015 = ZonedDateTime(2015, 1, 1, tz"Asia/Seoul")
        Jan_2015_hours = collect(Jan_2015:Hour(1):Jan_2015 + Day(30))
        len_df = length(Jan_2015_hours)

        df = DataFrame(
            date = Jan_2015_hours,
            A = collect(         1:  len_df),
            B = collect(  len_df+1:2*len_df),
            C = collect(2*len_df+1:3*len_df)
        )

        idx = 1:sample_size
        X, Y = make_input_LSTM(df, :C, [idx], [:A, :B], sample_size, output_size)

        #=
        output should be
        X = [[1,...,24] [len_df,...,len_df+24]]
        Y = [2*len_df,...,2*len_df+12]
        =#

        # back to cpu for test
        X = X |> cpu
        Y = Y |> cpu

        @test_broken size(X) == (1, sample_size, 2)
        @test_broken size(Y) == (1, output_size,)

        @test_broken X[1, :, :] == hcat([[collect(1:sample_size)] [collect(len_df + 1:len_df + sample_size)]]...)
        @test_broken Y[1, :] == collect((sample_size + 2*len_df + 1):(sample_size + 2*len_df + output_size))
    end

    @testset "is_sparse_Y_only_missings" begin
        pairs = [
            [ 5, 6, 7, 8, 9,10],
            [11,12,13,14,15,missing],
            [17,18,19,20,missing,missing],
            [23,24,25,missing,missing,missing],
            [29,30,missing,missing,missing,missing]]

        m_ratio = 0.5
        @test is_sparse_Y(pairs[1], m_ratio) == false
        @test is_sparse_Y(pairs[2], m_ratio) == false
        @test is_sparse_Y(pairs[3], m_ratio) == false
        @test is_sparse_Y(pairs[4], m_ratio) == true
        @test is_sparse_Y(pairs[5], m_ratio) == true

        m_ratio = 0.3
        @test is_sparse_Y(pairs[1], m_ratio) == false
        @test is_sparse_Y(pairs[2], m_ratio) == false
        @test is_sparse_Y(pairs[3], m_ratio) == true
        @test is_sparse_Y(pairs[4], m_ratio) == true
        @test is_sparse_Y(pairs[5], m_ratio) == true

    end

    @testset "is_sparse_Y_missings_and_zeros" begin
        pairs = [
            [ 5, 6, 7, 8, 9,10],
            [11,12,13,14,0.0,missing],
            [17,18,0.0,0.0,missing,missing],
            [23,24,0.0,missing,missing,missing],
            [29,30,missing,missing,missing,missing]]

        m_ratio = 0.5
        @test is_sparse_Y(pairs[1], m_ratio) == false
        @test is_sparse_Y(pairs[2], m_ratio) == false
        @test is_sparse_Y(pairs[3], m_ratio) == true
        @test is_sparse_Y(pairs[4], m_ratio) == true
        @test is_sparse_Y(pairs[5], m_ratio) == true
    end
end
=#
