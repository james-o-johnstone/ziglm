const std = @import("std");
const ggml = @import("ggml.zig");

const MAX_KEY_LEN = 65535;
const MAX_TENSOR_NAME_LEN = 64;
const MAX_VALUE_STRING_LEN = 1 << 24;
const MAX_ARRAY_LEN = 1 << 24;
const MAX_DIMENSIONS = 8;

const GGUFParseError = error{
    ArrayTooLong,
    BufferTooSmall,
    EndOfStream,
    InvalidEnumTag,
    InvalidValueType,
    OutOfMemory,
    ReadFailed,
    StringTooLong,
};

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
    type: ggml.GGMLType,
    // The offset of the tensor's data in this file in bytes.
    //
    // This offset is relative to `tensor_data`, not to the start
    // of the file, to make it easier for writers to write the file.
    // Readers should consider exposing this offset relative to the
    // file to make it easier to read the data.
    //
    // Must be a multiple of `ALIGNMENT`. That is, `align_offset(offset) == offset`.
    offset: u64,

    data: []const u8,

    fn parse(reader: *std.Io.Reader, allocator: std.mem.Allocator) !Tensor {
        const name = try GgufString.parse(allocator, reader, MAX_TENSOR_NAME_LEN);
        errdefer name.deinit(allocator);

        const n_dimensions = try reader.takeInt(u32, .little);
        if (n_dimensions > MAX_DIMENSIONS) return error.TooManyDimensions;

        const dimensions = try allocator.alloc(u64, n_dimensions);
        errdefer allocator.free(dimensions);

        for (dimensions) |*dim| {
            dim.* = try reader.takeInt(u64, .little);
        }

        const tensor_type = try reader.takeEnum(ggml.GGMLType, .little);
        const offset = try reader.takeInt(u64, .little);

        return Tensor{
            .name = name,
            .n_dimensions = n_dimensions,
            .dimensions = dimensions,
            .type = tensor_type,
            .offset = offset,
            .data = &.{}, // needs to point to the offset in the file
        };
    }

    pub fn deinit(self: *Tensor, allocator: std.mem.Allocator) void {
        self.name.deinit(allocator);
        allocator.free(self.dimensions);
    }

    pub fn byte_len(self: *Tensor) usize {
        var elements: u64 = 1;
        for (self.dimensions) |d| elements *= d;

        const trait = ggml.traits(self.type);
        const n_blocks = elements / trait.blck_size;
        return n_blocks * trait.type_size;
    }
};
// zig fmt: on

const GgufString = struct {
    len: u64,
    data: []const u8,

    fn parse(allocator: std.mem.Allocator, reader: *std.Io.Reader, max_len: u64) !GgufString {
        const len = try reader.takeInt(u64, .little);
        if (len > max_len) return error.StringTooLong;

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
        if (len > MAX_ARRAY_LEN) return error.ArrayTooLong;

        const arr = try allocator.alloc(GgufMetadataValue, len);
        var parsed: usize = 0;
        errdefer {
            for (arr[0..parsed]) |value| value.deinit(allocator);
            allocator.free(arr);
        }

        for (arr) |*value| {
            value.* = try GgufMetadataValue.parse(reader, value_type, allocator);
            parsed += 1;
        }

        return GgufArray{ .type = value_type, .len = len, .array = arr };
    }

    pub fn deinit(self: *const GgufArray, allocator: std.mem.Allocator) void {
        for (self.array) |value| {
            value.deinit(allocator);
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
                const value: f32 = @bitCast(try reader.takeInt(u32, .little));
                return GgufMetadataValue{ .GGUF_METADATA_VALUE_TYPE_FLOAT32 = value };
            },
            .GGUF_METADATA_VALUE_TYPE_BOOL => {
                const value = try reader.takeByte();
                return GgufMetadataValue{ .GGUF_METADATA_VALUE_TYPE_BOOL = value != 0 };
            },
            .GGUF_METADATA_VALUE_TYPE_STRING => {
                const value = try GgufString.parse(allocator, reader, MAX_VALUE_STRING_LEN);
                return GgufMetadataValue{ .GGUF_METADATA_VALUE_TYPE_STRING = value };
            },
            .GGUF_METADATA_VALUE_TYPE_ARRAY => {
                const value = try GgufArray.parse(reader, allocator);
                return GgufMetadataValue{ .GGUF_METADATA_VALUE_TYPE_ARRAY = value };
            },
            .GGUF_METADATA_VALUE_TYPE_UINT64 => {
                const value = try reader.takeInt(u64, .little);
                return GgufMetadataValue{ .GGUF_METADATA_VALUE_TYPE_UINT64 = value };
            },
            .GGUF_METADATA_VALUE_TYPE_INT64 => {
                const value = try reader.takeInt(i64, .little);
                return GgufMetadataValue{ .GGUF_METADATA_VALUE_TYPE_INT64 = value };
            },
            .GGUF_METADATA_VALUE_TYPE_FLOAT64 => {
                const value: f64 = @bitCast(try reader.takeInt(u64, .little));
                return GgufMetadataValue{ .GGUF_METADATA_VALUE_TYPE_FLOAT64 = value };
            },
        }
    }

    pub fn deinit(self: *const GgufMetadataValue, allocator: std.mem.Allocator) void {
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
    key: GgufString,
    value_type: GgufMetadataValueType,
    value: GgufMetadataValue,

    fn parse(reader: *std.Io.Reader, allocator: std.mem.Allocator) !GgufMetadataKvT {
        const key = try GgufString.parse(allocator, reader, MAX_KEY_LEN);
        errdefer key.deinit(allocator);

        std.debug.print("k: {s}\n", .{key.data});

        const value_type = try reader.takeEnum(GgufMetadataValueType, .little);
        std.debug.print("about to read meta value, type={any}\n", .{value_type});
        const value: GgufMetadataValue = try GgufMetadataValue.parse(reader, value_type, allocator);

        return GgufMetadataKvT{ .key = key, .value_type = value_type, .value = value };
    }

    pub fn deinit(self: *GgufMetadataKvT, allocator: std.mem.Allocator) void {
        self.key.deinit(allocator);
        self.value.deinit(allocator);
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

    fn initFromMmap(allocator: std.mem.Allocator, memory: []const u8) !Gguf {
        var r = std.Io.Reader.fixed(memory);

        const m = try r.takeArray(4);
        const magic = m.*;
        const version = try r.takeInt(u32, .little);

        if (!std.mem.eql(u8, &magic, Gguf.MAGIC)) return error.InvalidFile;
        if (version != Gguf.VERSION) return error.UnsupportedVersion;

        const tensor_count = try r.takeInt(u64, .little);
        const metadata_kv_count = try r.takeInt(u64, .little);

        var metadata = std.StringHashMap(GgufMetadataKvT).init(allocator);
        errdefer {
            var it = metadata.iterator();
            while (it.next()) |entry| {
                entry.value_ptr.*.deinit(allocator);
            }
            metadata.deinit();
        }
        std.debug.print("about to read metadata\n", .{});
        for (0..metadata_kv_count) |_| {
            var kv = try GgufMetadataKvT.parse(&r, allocator);
            errdefer kv.deinit(allocator);
            std.debug.print("Inserting key: {s}\n", .{kv.key.data});
            const slot = try metadata.getOrPut(kv.key.data);
            if (slot.found_existing) return error.DuplicateKey;
            slot.value_ptr.* = kv;
        }

        var tensors = std.ArrayList(Tensor).empty;
        errdefer {
            for (tensors.items) |*t| t.deinit(allocator);
            tensors.deinit(allocator);
        }
        for (0..tensor_count) |_| {
            var tensor = try Tensor.parse(&r, allocator);
            errdefer tensor.deinit(allocator);
            try tensors.append(allocator, tensor);
        }

        const alignment: u32 = if (metadata.get("general.alignment")) |kv|
            kv.value.GGUF_METADATA_VALUE_TYPE_UINT32
        else
            32;

        const data_start = std.mem.alignForward(usize, r.seek, alignment);
        for (tensors.items) |*tensor| {
            const start = data_start + tensor.offset;
            const len = tensor.byte_len();
            if (start + len > memory.len) return error.TensorOutOfBounds;
            tensor.data = memory[start..][0..len];
        }

        return Gguf{ .magic = magic, .version = version, .tensor_count = tensor_count, .metadata_kv_count = metadata_kv_count, .metadata = metadata, .tensors = tensors };
    }

    fn initFromFile(io: std.Io, allocator: std.mem.Allocator, file: std.Io.File) !Gguf {
        var buf: [4096]u8 = undefined;
        var reader = file.readerStreaming(io, &buf);
        var r = &reader.interface;

        const m = try r.takeArray(4);
        const magic = m.*;
        const version = try r.takeInt(u32, .little);

        if (!std.mem.eql(u8, &magic, Gguf.MAGIC)) return error.InvalidFile;
        if (version != Gguf.VERSION) return error.UnsupportedVersion;

        const tensor_count = try r.takeInt(u64, .little);
        const metadata_kv_count = try r.takeInt(u64, .little);

        var metadata = std.StringHashMap(GgufMetadataKvT).init(allocator);
        errdefer {
            var it = metadata.iterator();
            while (it.next()) |entry| {
                entry.value_ptr.*.deinit(allocator);
            }
            metadata.deinit();
        }
        std.debug.print("about to read metadata\n", .{});
        for (0..metadata_kv_count) |_| {
            var kv = try GgufMetadataKvT.parse(r, allocator);
            errdefer kv.deinit(allocator);
            std.debug.print("Inserting key: {s}\n", .{kv.key.data});
            const slot = try metadata.getOrPut(kv.key.data);
            if (slot.found_existing) return error.DuplicateKey;
            slot.value_ptr.* = kv;
        }

        var tensors = std.ArrayList(Tensor).empty;
        errdefer {
            for (tensors.items) |*t| t.deinit(allocator);
            tensors.deinit(allocator);
        }
        for (0..tensor_count) |_| {
            var tensor = try Tensor.parse(r, allocator);
            errdefer tensor.deinit(allocator);
            try tensors.append(allocator, tensor);
        }
        return Gguf{ .magic = magic, .version = version, .tensor_count = tensor_count, .metadata_kv_count = metadata_kv_count, .metadata = metadata, .tensors = tensors };
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
            entry.value_ptr.*.deinit(allocator);
        }
        self.metadata.deinit();
        for (self.tensors.items) |*item| {
            item.deinit(allocator);
        }
        self.tensors.deinit(allocator);
    }
};

pub fn load(io: std.Io, allocator: std.mem.Allocator, filename: []const u8) !void {
    var file = std.Io.Dir.cwd().openFile(io, filename, .{}) catch |err| {
        std.debug.print("Failed to open file: {s}\n", .{filename});
        return err;
    };
    defer file.close(io);

    const len: usize = @intCast(try file.length(io));
    var mm = try file.createMemoryMap(io, .{
        .len = len,
        .protection = .{ .read = true }, // read only
        .populate = false,
    });
    defer mm.destroy(io);
    const bytes = mm.memory;
    var gguf = try Gguf.initFromMmap(allocator, bytes);
    defer gguf.deinit(allocator);
    gguf.print();
}

test "load" {
    try load(std.testing.io, std.testing.allocator, "test.txt");
}
