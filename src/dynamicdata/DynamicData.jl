"""
Module to store dynamic data. Dynamic data in general is comprised of a set of data points, each of which is associated to
a timestamp. Both the type of data and timestamp can be chosen freely by using the appropriate parameterization.
"""
module DynamicData
using DataStructures
import Base: <, <=, ==, length, isempty, start, next, done, size, eltype, get, get!, getindex, endof

# delegate macro by John Myles White
macro delegate(source, targets)
    typename = esc(source.args[1])
    fieldname = source.args[2].args[1]
    funcnames = targets.args
    n = length(funcnames)
    fdefs = Array(Any, n)
    for i in 1:n
        funcname = esc(funcnames[i])
        fdefs[i] = quote
                     ($funcname)(a::($typename), args...) =
                       ($funcname)(a.$fieldname, args...)
                   end
    end
    return Expr(:block, fdefs...)
end

"""
DataPoint type. A DataPoint consists of the actual data of type D and an associated timestamp of type T.
"""
type DataPoint{D, T}
	id::Int64
	timestamp::T
	value::D
	function DataPoint()
		new(-1, D(), T())
	end
	function DataPoint{D, T}(_t::T, _v::D)
		new(-1, _t, _v)
	end
end
DataPoints{D,T} = Vector{DataPoint{D,T}}

"""
The DynamicDataSet consists of a set of DataPoints of a common value and timestamp types.
Data structures and methods to access DataPoints directly or via their timestamps are provided.
Basic functionality like length and indexing are inherited from Base.
"""
type DynamicDataSet{D, T}
	time_dict::Dict{Int64, T}
	reverse_time_dict::Dict{T, Int64}
	data_points::DataPoints{D, T}
	data_indexes::Dict{T, DataPoints{D,T}}

	function DynamicDataSet()
		time_dict = Dict{Int64, T}()
		reverse_time_dict = Dict{T, Int64}()
		data_points = DataPoints{D, T}()
		data_indexes = Dict{T, DataPoints{D,T}}()
		new(time_dict, reverse_time_dict, data_points, data_indexes)
	end

	function DynamicDataSet(data::Array{D, 1}, timestamps::Array{T, 1})
		time_dict = Dict{Int64, T}()
		reverse_time_dict = Dict{T, Int64}()
		data_points = DataPoints{D, T}()
		data_indexes = Dict{T, DataPoints{D,T}}()
		dynDat = new(time_dict, reverse_time_dict, data_points, data_indexes)

		for (i, dp) in enumerate(data)
			add(dynDat, dp, timestamps[i], false)
		end
		rebuild_indexes(dynDat)
		dynDat
	end
end

@delegate DynamicDataSet.data_points [ get, get!, getindex, length, isempty, eltype, endof, size, start, next, done]

"""
Creates and adds a new DataPoint to the DynamicDataSet based on the provided data of type D, and the associated timestamp of type T.
The parameter ri indicates whether to rebuild the indexes after the insert operation. If you are filling a new dataset with the add method,
set ri to false and explicitly call rebuild_indexes after all insert operations are done for better performance.
"""
function add{D,T}(d::DynamicDataSet{D,T}, data::D, timestamp::T, ri = true)
	#check whether we already have data for the provided timestamp
	if !haskey(d.reverse_time_dict, timestamp)
		#if not, append the timestamp to the reverse_time_dict
		d.reverse_time_dict[timestamp] = length(d.reverse_time_dict) + 1
		d.data_indexes[timestamp] = DataPoints{D,T}()
	end
	#create data point
	dp = DataPoint{D,T}(timestamp, data)
	push!(d.data_points, dp)
	push!(d.data_indexes[timestamp], dp)
	if ri
		rebuild_indexes(d)
	end
	dp
end

function remove!{D,T}(d::DynamicDataSet{D,T}, dp::DataPoint{D,T})
	dp_id = findin(d.data_points, [dp])
	remove!(d, dp_id[1])
end

function remove!{D,T}(d::DynamicDataSet{D,T}, dp_id::Int64)
	d.data_indexes[d.data_points[dp_id].timestamp] = setdiff(d.data_indexes[d.data_points[dp_id].timestamp], [d.data_points[dp_id]])
	deleteat!(d.data_points, dp_id)
end

"""
Rebuilds the indexes after insertion operations.
"""
function rebuild_indexes(d::DynamicDataSet)
	#get all timestamps
	used_timestamps = get_unique_timestamps(d)
	#update the time_dict structure
	for (i, ts) in enumerate(used_timestamps)
		setindex!(d.reverse_time_dict, i, ts)
		setindex!(d.time_dict, ts, i)
	end
end

function convert_datapoint_array{T}(dps::Array{DataPoint{Array{Float64,1},T}, 1})
	ret = zeros(length(dps), length(dps[1].value))
	for (i, dp) in enumerate(dps)
		ret[i,:] = dp.value
	end
	ret
end

"""
Gets all unique timestamps (sorted).
"""
function get_unique_timestamps(d::DynamicData.DynamicDataSet)
	sort(collect(keys(d.reverse_time_dict)))
end

"""
Timestamp getters returning a set of timestamps based on the full data set, a datapoint subset defined by their
IDs in the data set or by an array of elements of type DataPoint.
"""
get_timestamps(d::DynamicDataSet) = map((x) -> x.timestamp, d.data_points)
get_timestamps_for_subset(d::DynamicDataSet, ids::Vector{Int64}) = map((x) -> x.timestamp, d.data_points[ids])
get_timestamps_for_subset{D,T}(pts::Vector{DataPoint{D,T}}) = map((x)->x.timestamp, pts)
get_unique_timestamps_for_subset(d::DynamicDataSet, ids::Vector{Int64}) = unique(get_timestamps_for_subset(d,ids))
get_unique_timestamps_for_subset{D,T}(pts::Vector{DataPoint{D,T}}) = unique(get_timestamps_for_subset(pts))

"""
Gets the timestamp value based on its index.
"""
get_timestamp(d::DynamicDataSet, ti::Int64) = d.time_dict[ti]

"""
Gets the timestamp index based on its value.
"""
get_timestamp_index{T}(d::DynamicDataSet, ts::T) = d.reverse_time_dict[ts]

"""
Gets timestamp index for a given datapoint index.
"""
get_timestamp_index_for_datapoint(d::DynamicDataSet, dp_index::Int64) = d.reverse_time_dict[d.data_points[dp_index].timestamp]

"""
Gets the timestamp indexes of a given index set of data points.
"""
function get_timestamp_indexes_for_subset(d::DynamicDataSet, ids::Vector{Int64})
	timestamp_ids = Array(Int64, length(ids))
	for (i,dp) in enumerate(d[ids])
		timestamp_ids[i] = d.reverse_time_dict[dp.timestamp]
	end
	IntSet(timestamp_ids)
end

"""
Gets the timestamp indexes for a given array of elements of type DataPoint.
"""
function get_timestamp_indexes_for_subset{D,T}(d::DynamicDataSet{D,T}, dps::Vector{DataPoint{D,T}})
	timestamp_ids = Vector{Int64}(length(dps))
	for (i, dp) in enumerate(dps)
		timestamp_ids[i] = d.reverse_time_dict[dp.timestamp]
	end
	IntSet(timestamp_ids)
end


"""
Gets the datapoints associated with a given timestamp.
"""
get_datapoints_for_timestamp{T}(d::DynamicDataSet, ts::T) = d.data_indexes[ts]

"""
Get the datapoint values for an array of elements of type DataPoint.
"""
get_datapoint_values{D,T}(pts::Array{DataPoint{D,T},1}) = map((x)->x.value, pts)

"""
Gets the datapoints associated with the given timestamp based on the provided array of datapoints.
"""
function get_datapoints_for_timestamp_in_subset{D,T}(dps::Array{DataPoint{D,T}, 1}, ts::T)
	ret = Array(DataPoint, 0)
	for dp in dps
		if dp.timestamp == ts
			ret = [ret;dp]
		end
	end
	ret
end

"""
Gets the indexes of datapoints that are associated with a given timestamp based on the provided set of datapoints.
"""
function get_datapoint_idx_for_timestamp_in_subset{D,T}(dps::Array{DataPoint{D,T}, 1}, ts::T)
	ret = Int64[]
	for (i, dp) in enumerate(dps)
		if dp.timestamp == ts
			ret = [ret;i]
		end
	end
	ret
end

"""
Returns an array mapping the timestamp indexes to the number of datapoints associated with them.
"""
function get_datacounts(d::DynamicDataSet)
	sorted_time_ids = sort(collect(keys(d.time_dict)))
	ret = Array(Int64, length(sorted_time_ids))
	for t_id in sorted_time_ids
		ret[t_id] = length(d.data_indexes[d.time_dict[t_id]])
	end
	ret
end

"""
Returns an array mapping the timestamps used in the given subset (provided as DataPoint indexes)
to the number of datapoints associated with them in the subset.
"""
function get_datacounts_for_subset(d::DynamicDataSet, ids::Array{Int64, 1})
	ret = zeros(Int64, length(d.time_dict))
	for id in ids
		t_i = get_timestamp_index_for_datapoint(d, id)
		ret[t_i] +=1
	end
	ret
end

"""
Returns an array mapping the timestamps used in the given subset (provided as an array of DataPoint elements)
to the number of datapoints associated with them in the subset.
"""
function get_datacounts_for_subset{D,T}(d::DynamicDataSet{D,T}, dps::Array{DataPoint{D,T}, 1})
	ret = zeros(Int64, length(d.time_dict))
	for dp in dps
		t_i = get_timestamp_index(d, dp.timestamp)
		ret[t_i] +=1
	end
	ret
end

export DynamicDataSet, DataPoint, add

end
