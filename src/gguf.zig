const std = @import("std");

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

// zig fmt: off
const GGMLType = enum(u32) {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    // 4, 5: Q4_2, Q4_3 — support removed
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S = 19,
    GGML_TYPE_IQ4_NL = 20,
    GGML_TYPE_IQ3_S = 21,
    GGML_TYPE_IQ2_S = 22,
    GGML_TYPE_IQ4_XS = 23,
    GGML_TYPE_I8 = 24,
    GGML_TYPE_I16 = 25,
    GGML_TYPE_I32 = 26,
    GGML_TYPE_I64 = 27,
    GGML_TYPE_F64 = 28,
    GGML_TYPE_IQ1_M = 29,
    GGML_TYPE_BF16 = 30,
    // 31-33: Q4_0_4_4, Q4_0_4_8, Q4_0_8_8 — removed from gguf files
    GGML_TYPE_TQ1_0 = 34,
    GGML_TYPE_TQ2_0 = 35,
    // 36-38: IQ4_NL_4_4, IQ4_NL_4_8, IQ4_NL_8_8 — removed
    GGML_TYPE_MXFP4 = 39,
    GGML_TYPE_COUNT = 40,
};
// zig fmt: on

// zig fmt: off
const Tensor = struct {
    // The name of the tensor. It is a standard GGUF string, with the caveat that
    // it must be at most 64 bytes long.
    name: GgufString,
    // The number of dimensions in the tensor.
    // Currently at most 4, but this may change in the future.
    n_dimensions: u32,
    // The dimensions of the tensor.
    dimensions: []const u64,
    // The type of the tensor.
    type: GGMLType,
    // The offset of the tensor's data in this file in bytes.
    //
    // This offset is relative to `tensor_data`, not to the start
    // of the file, to make it easier for writers to write the file.
    // Readers should consider exposing this offset relative to the
    // file to make it easier to read the data.
    //
    // Must be a multiple of `ALIGNMENT`. That is, `align_offset(offset) == offset`.
    offset: u64,

    fn parse(reader: *std.Io.Reader, allocator: std.mem.Allocator) !Tensor {
        const name = try GgufString.parse(allocator, reader);
        errdefer name.deinit(allocator);

        const n_dimensions = try reader.takeInt(u32, .little);

        const dimensions = try allocator.alloc(u64, n_dimensions);
        errdefer allocator.free(dimensions);

        for (dimensions) |*dim| {
            dim.* = try reader.takeInt(u64, .little);
        }

        const tensor_type = try reader.takeEnum(GGMLType, .little);
        const offset = try reader.takeInt(u64, .little);

        return Tensor{
            .name = name,
            .n_dimensions = n_dimensions,
            .dimensions = dimensions,
            .type = tensor_type,
            .offset = offset,
        };
    }

    pub fn deinit(self: *Tensor, allocator: std.mem.Allocator) void {
        self.name.deinit(allocator);
        allocator.free(self.dimensions);
    }
};
// zig fmt: on

const GgufString = struct {
    len: u64,
    data: []const u8,

    fn parse(allocator: std.mem.Allocator, reader: *std.Io.Reader) !GgufString {
        const len = try reader.takeInt(u64, .little);
        if (len > 65535) return error.StringTooLong;

        const data = try allocator.alloc(u8, len);
        errdefer allocator.free(data);

        try reader.readSliceAll(data);

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

        for (arr) |*value| {
            value.* = try GgufMetadataValue.parse(reader, value_type, allocator);
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

    magic: [4]u8,
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
    metadata: std.StringHashMap(GgufMetadataKvT),
    tensors: std.ArrayList(Tensor),

    fn initFromFile(io: std.Io, allocator: std.mem.Allocator, file: std.Io.File) !Gguf {
        var buf: [4096]u8 = undefined;
        var reader = file.readerStreaming(io, &buf);
        var r = &reader.interface;

        const magic = try r.takeArray(4);
        const magic_bytes = magic.*;
        const version = try r.takeInt(u32, .little);

        if (!std.mem.eql(u8, &magic_bytes, Gguf.MAGIC)) return error.InvalidFile;
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
        std.debug.print("about to read metadata\n", .{});
        for (0..metadata_kv_count) |_| {
            const k = try GgufString.parse(allocator, r);
            std.debug.print("k: {s}\n", .{k.data});
            const v = try GgufMetadataKvT.parse(r, allocator);
            std.debug.print("Inserting key: {s}\n", .{k.data});
            try metadata.put(k.data, v);
        }

        var tensors = std.ArrayList(Tensor).empty;
        for (0..tensor_count) |_| {
            const tensor = try Tensor.parse(r, allocator);
            try tensors.append(allocator, tensor);
        }
        return Gguf{ .magic = magic_bytes, .version = version, .tensor_count = tensor_count, .metadata_kv_count = metadata_kv_count, .metadata = metadata, .tensors = tensors };
    }

    fn print(self: *Gguf) void {
        std.debug.print("Magic number: {s}\n", .{self.magic[0..]});
        std.debug.print("Version: {x}\n", .{self.version});
        std.debug.print("Tensor count: {d}\n", .{self.tensor_count});
        std.debug.print("Metadata key-value count: {d}\n", .{self.metadata_kv_count});
        for (self.tensors.items) |tensor| {
            std.debug.print("Tensor: {s}, shape: {any}, type: {s}\n", .{ tensor.name.data, tensor.dimensions, @tagName(tensor.type) });
        }
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
        for (self.tensors.items) |*item| {
            item.deinit(allocator);
        }
        self.tensors.deinit(allocator);
    }
};

pub fn load(io: std.Io, allocator: std.mem.Allocator, filename: []const u8) !void {
    var file = std.Io.Dir.cwd().openFile(io, filename, .{}) catch {
        std.debug.print("Failed to open file: {s}\n", .{filename});
        return;
    };
    var gguf = try Gguf.initFromFile(io, allocator, file);
    defer gguf.deinit(allocator);
    gguf.print();
    defer file.close(io);
}

test "load" {
    try load(std.testing.io, std.testing.allocator, "test.txt");
}
