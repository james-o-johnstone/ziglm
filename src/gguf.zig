const std = @import("std");

const GGUFError = error{
    InvalidFile,
    UnsupportedVersion,
};

const GGUFParseError = error{
    BufferTooSmall,
    EndOfStream,
    InvalidEnumTag,
    InvalidValueType,
    OutOfMemory,
    ReadFailed,
    StringTooLong,
    Unimplemented,
};

const Tensor = struct { name: []const u8, n_dimensions: u32, dimensions: u64, type: u32, offset: u64 };

const GgufString = struct {
    len: u64,
    data: []const u8,

    fn parse(allocator: std.mem.Allocator, reader: *std.Io.Reader) !GgufString {
        const len = try reader.takeInt(u64, .little);
        if (len > 65535) return error.StringTooLong;

        var data = try allocator.alloc(u8, len);
        errdefer allocator.free(data);

        var i: usize = 0;
        while (i < len) : (i += 1) {
            const byte = try reader.takeByte();
            data[i] = byte;
        }

        return GgufString{ .len = len, .data = data };
    }

    pub fn deinit(self: *const GgufString, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
    }
};
const GgufArray = struct {
    // Any value type is valid, including arrays.
    type: GgufMetadataValueType,
    // Number of elements, not bytes
    len: u64,
    // The array of values.
    array: []const GgufMetadataValue,

    fn parse(reader: *std.Io.Reader, allocator: std.mem.Allocator) GGUFParseError!GgufArray {
        const value_type = try reader.takeEnum(GgufMetadataValueType, .little);
        const len = try reader.takeInt(u64, .little);

        const arr = try allocator.alloc(GgufMetadataValue, len);
        errdefer allocator.free(arr);

        var i: u64 = 0;
        while (i < len) : (i += 1) {
            arr[i] = try GgufMetadataValue.parse(reader, value_type, allocator);
        }

        return GgufArray{ .type = value_type, .len = len, .array = arr };
    }

    pub fn deinit(self: *const GgufArray, allocator: std.mem.Allocator) void {
        for (self.array) |value| {
            switch (value) {
                .GGUF_METADATA_VALUE_TYPE_STRING => {
                    value.GGUF_METADATA_VALUE_TYPE_STRING.deinit(allocator);
                },
                .GGUF_METADATA_VALUE_TYPE_ARRAY => {
                    value.GGUF_METADATA_VALUE_TYPE_ARRAY.deinit(allocator);
                },
                else => {},
            }
        }
        allocator.free(self.array);
    }
};

const GgufMetadataValueType = enum(u32) {
    // The value is a 8-bit unsigned integer.
    GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
    // The value is a 8-bit signed integer.
    GGUF_METADATA_VALUE_TYPE_INT8 = 1,
    // The value is a 16-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
    // The value is a 16-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT16 = 3,
    // The value is a 32-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
    // The value is a 32-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT32 = 5,
    // The value is a 32-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    GGUF_METADATA_VALUE_TYPE_BOOL = 7,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    GGUF_METADATA_VALUE_TYPE_STRING = 8,
    // The value is an array of other values, with the length and type prepended.
    ///
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
    // The value is a 64-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
    // The value is a 64-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT64 = 11,
    // The value is a 64-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
};

// zig fmt: off
const GgufMetadataValue = union(GgufMetadataValueType) { 
    GGUF_METADATA_VALUE_TYPE_UINT8: u8, 
    GGUF_METADATA_VALUE_TYPE_INT8: i8, 
    GGUF_METADATA_VALUE_TYPE_UINT16: u16, 
    GGUF_METADATA_VALUE_TYPE_INT16: i16, 
    GGUF_METADATA_VALUE_TYPE_UINT32: u32, 
    GGUF_METADATA_VALUE_TYPE_INT32: i32, 
    GGUF_METADATA_VALUE_TYPE_FLOAT32: f32, 
    GGUF_METADATA_VALUE_TYPE_BOOL: bool,
    GGUF_METADATA_VALUE_TYPE_STRING: GgufString,
    GGUF_METADATA_VALUE_TYPE_ARRAY: GgufArray,
    GGUF_METADATA_VALUE_TYPE_UINT64: u64, 
    GGUF_METADATA_VALUE_TYPE_INT64: i64, 
    GGUF_METADATA_VALUE_TYPE_FLOAT64: f64, 

    fn parse(reader: *std.Io.Reader, value_type: GgufMetadataValueType, allocator: std.mem.Allocator) GGUFParseError!GgufMetadataValue {
        switch (value_type) {
            .GGUF_METADATA_VALUE_TYPE_UINT8 => {
                const value = try reader.takeInt(u8, .little);
                return GgufMetadataValue{ .GGUF_METADATA_VALUE_TYPE_UINT8 = value };
            },
            .GGUF_METADATA_VALUE_TYPE_INT8 => {
                const value = try reader.takeInt(i8, .little);
                return GgufMetadataValue{ .GGUF_METADATA_VALUE_TYPE_INT8 = value };
            },
            .GGUF_METADATA_VALUE_TYPE_UINT16 => {
                const value = try reader.takeInt(u16, .little);
                return GgufMetadataValue{ .GGUF_METADATA_VALUE_TYPE_UINT16 = value };
            },
            .GGUF_METADATA_VALUE_TYPE_INT16 => {
                const value = try reader.takeInt(i16, .little);
                return GgufMetadataValue{ .GGUF_METADATA_VALUE_TYPE_INT16 = value };
            },
            .GGUF_METADATA_VALUE_TYPE_UINT32 => {
                const value = try reader.takeInt(u32, .little);
                return GgufMetadataValue{ .GGUF_METADATA_VALUE_TYPE_UINT32 = value };
            },
            .GGUF_METADATA_VALUE_TYPE_INT32 => {
                const value = try reader.takeInt(i32, .little);
                return GgufMetadataValue{ .GGUF_METADATA_VALUE_TYPE_INT32 = value };
            },
            .GGUF_METADATA_VALUE_TYPE_FLOAT32 => {
                const buf = try reader.take(4);
                const value = std.mem.bytesToValue(f32, buf[0..4]);
                return GgufMetadataValue{ .GGUF_METADATA_VALUE_TYPE_FLOAT32 = value };
            },
            .GGUF_METADATA_VALUE_TYPE_BOOL => {
                const value = try reader.takeByte();
                return GgufMetadataValue{ .GGUF_METADATA_VALUE_TYPE_BOOL = value != 0 };
            },
            .GGUF_METADATA_VALUE_TYPE_STRING => {
                const value = try GgufString.parse(allocator, reader);
                return GgufMetadataValue{ .GGUF_METADATA_VALUE_TYPE_STRING = value };
            },
            .GGUF_METADATA_VALUE_TYPE_ARRAY => {
                const value = try GgufArray.parse(reader, allocator);
                return GgufMetadataValue{ .GGUF_METADATA_VALUE_TYPE_ARRAY = value };
            },
            else => {
                return error.Unimplemented;
            },
        }

        // do i need to switch case the union types or is there a beter way?

        


        //var len_bytes: [8]u8 = undefined;
        //_ = try reader.readSliceShort(&len_bytes);
        //const len = std.mem.bytesAsValue(u64, len_bytes[0..8]).*;

        //var data_bytes: []u8 = undefined;
        //_ = try reader.readSliceShort(&data_bytes[0..len]);

        //return GgufString{ .len = len, .data = data_bytes[0..len] };
        return error.Unimplemented;
    }

    pub fn deinit(self: *GgufMetadataValue, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .GGUF_METADATA_VALUE_TYPE_STRING => {
                self.GGUF_METADATA_VALUE_TYPE_STRING.deinit(allocator);
            },
            .GGUF_METADATA_VALUE_TYPE_ARRAY => {
                self.GGUF_METADATA_VALUE_TYPE_ARRAY.deinit(allocator);
            },
            else => {},
        }
    }
};
// zig fmt: on

const GgufMetadataKvT = struct {
    // The key of the metadata. It is a standard GGUF string, with the following caveats:
    // - It must be a valid ASCII string.
    // - It must be a hierarchical key, where each segment is `lower_snake_case` and separated by a `.`.
    // - It must be at most 2^16-1/65535 bytes long.
    // Any keys that do not follow these rules are invalid.
    value_type: GgufMetadataValueType,
    value: GgufMetadataValue,

    fn parse(reader: *std.Io.Reader, allocator: std.mem.Allocator) !GgufMetadataKvT {
        const value_type = try reader.takeEnum(GgufMetadataValueType, .little);

        std.debug.print("about to read meta value, type={any}\n", .{value_type});
        const value: GgufMetadataValue = try GgufMetadataValue.parse(reader, value_type, allocator);

        return GgufMetadataKvT{ .value_type = value_type, .value = value };
    }

    pub fn deinit(self: *GgufMetadataKvT, allocator: std.mem.Allocator) void {
        switch (self.value_type) {
            .GGUF_METADATA_VALUE_TYPE_STRING => {
                self.value.deinit(allocator);
            },
            .GGUF_METADATA_VALUE_TYPE_ARRAY => {
                self.value.deinit(allocator);
            },
            else => {},
        }
    }
};

const Gguf = struct {
    const MAGIC = "GGUF";
    const VERSION = 3;

    magic: []const u8,
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
    metadata: std.StringHashMap(GgufMetadataKvT),
    tensors: []Tensor,

    fn initFromFile(io: std.Io, allocator: std.mem.Allocator, file: std.Io.File) !Gguf {
        var buf: [4096]u8 = undefined;
        var reader = file.readerStreaming(io, &buf);
        var r = &reader.interface;

        const magic = try r.takeArray(4);
        var magic_array: [4]u8 = undefined;
        @memcpy(&magic_array, &magic[0..]);
        std.debug.print("Magic: {s}\n", .{magic_array[0..]});
        const version = try r.takeInt(u32, .little);

        if (!std.mem.eql(u8, magic[0..], Gguf.MAGIC)) return error.InvalidFile;
        if (version != Gguf.VERSION) return error.UnsupportedVersion;

        const tensor_count = try r.takeInt(u64, .little);
        const metadata_kv_count = try r.takeInt(u64, .little);

        var metadata = std.StringHashMap(GgufMetadataKvT).init(allocator);
        errdefer {
            var it = metadata.iterator();
            while (it.next()) |entry| {
                allocator.free(entry.key_ptr.*);
            }
            metadata.deinit();
        }
        var i: usize = 0;
        std.debug.print("about to read metadata\n", .{});
        while (i < metadata_kv_count) : (i += 1) {
            const k = try GgufString.parse(allocator, r);
            std.debug.print("k: {s}\n", .{k.data});
            const v = try GgufMetadataKvT.parse(r, allocator);

            //var string_bytes = std.mem.asBytes(&k);
            //_ = try r.readSliceShort(string_bytes[0..]);

            //var v: GgufString = undefined;
            //string_bytes = std.mem.asBytes(&v);
            //_ = try r.readSliceShort(string_bytes[0..]);

            //std.debug.print("Metadata: k:{any} v:{any}\n", .{ k.data, v });
            std.debug.print("Inserting key: {s}\n", .{k.data});
            try metadata.put(k.data, v);
        }

        //var tensors: []Tensor = std.ArrayList(Tensor).init(allocator).toSlice();
        //for (tensor_count) |i| {
        //    const name_len_bytes = try r.readBytes(4);
        //    const name_len = @intFromBytes(u32, name_len_bytes);
        //    const name = try r.readBytes(name_len);

        //    const n_dimensions_bytes = try r.readBytes(4);
        //    const n_dimensions = @intFromBytes(u32, n_dimensions_bytes);

        //    var dimensions = std.ArrayList(u64).init(allocator).toSlice();
        //    for (n_dimensions) |j| {
        //        const dim_bytes = try r.readBytes(8);
        //        dimensions.append(@intFromBytes(u64, dim_bytes));
        //    }

        //    const type_bytes = try r.readBytes(4);
        //    const type = @intFromBytes(u32, type_bytes);

        //    const offset_bytes = try r.readBytes(8);
        //    const offset = @intFromBytes(u64, offset_bytes);

        //    tensors.append(Tensor{
        //        .name = name,
        //        .n_dimensions = n_dimensions,
        //        .dimensions = dimensions[0..n_dimensions],
        //        .type = type,
        //        .offset = offset,
        //    });
        //}

        //return Gguf{
        //    .magic_number = magic_number,
        //    .version = version,
        //    .tensor_count = tensor_count,
        //    .metadata_kv_count = metadata_kv_count,
        //    .metadata = metadata,
        //    .tensors = tensors,
        //};
        return Gguf{ .magic = magic_array, .version = version, .tensor_count = tensor_count, .metadata_kv_count = metadata_kv_count, .metadata = metadata, .tensors = &.{} };
    }

    fn print(self: *Gguf) void {
        std.debug.print("Magic number: {s}\n", .{self.magic[0..]});
        std.debug.print("Version: {x}\n", .{self.version});
        std.debug.print("Tensor count: {d}\n", .{self.tensor_count});
        std.debug.print("Metadata key-value count: {d}\n", .{self.metadata_kv_count});
    }

    pub fn deinit(self: *Gguf, allocator: std.mem.Allocator) void {
        var it = self.metadata.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            switch (entry.value_ptr.*.value_type) {
                .GGUF_METADATA_VALUE_TYPE_STRING => {
                    entry.value_ptr.*.deinit(allocator);
                },
                .GGUF_METADATA_VALUE_TYPE_ARRAY => {
                    entry.value_ptr.*.deinit(allocator);
                },
                else => {},
            }
        }
        self.metadata.deinit();
    }
};

pub fn parse(filename: []const u8) !void {
    std.debug.print("{s}\n", .{filename});
}

pub fn load(io: std.Io, allocator: std.mem.Allocator, filename: []const u8) !void {
    var file = std.Io.Dir.cwd().openFile(io, filename, .{}) catch {
        std.debug.print("Failed to open file: {s}\n", .{filename});
        return;
    };
    //var gguf: Gguf = try Gguf.initFromFile(io, allocator, file);
    var gguf = try Gguf.initFromFile(io, allocator, file);
    defer gguf.deinit(allocator);
    gguf.print();
    defer file.close(io);
}

test "parse" {
    try parse("gguf.zig");
}

test "load" {
    try load(std.testing.io, std.testing.allocator, "test.txt");
}
