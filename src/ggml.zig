const std = @import("std");

// zig fmt: off
pub const GGMLType = enum(u32) {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q6_K = 14,
};
// zig fmt: on

// QK = number of values after dequantization
// QK_K = super-block size
const QK_K = 256;
const K_SCALE_SIZE = 12;

// 4-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight
const BlockQ4K = extern struct {
    d: f16, // super-block scale for quantized scales
    dmin: f16, // super-block scale for quantized mins
    scales: [K_SCALE_SIZE]u8, // scales and mins, quantized with 6 bits
    qs: [QK_K / 2]u8, // 4--bit quants
};

// 6-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 6.5625 bits per weight
const BlockQ6K = extern struct {
    ql: [QK_K / 2]u8, // quants, lower 4 bits
    qh: [QK_K / 4]u8, // quants, upper 2 bits
    scales: [QK_K / 16]i8, // scales, quantized with 8 bits
    d: f16, // super-block scale
};

const TypeTraits = struct { name: []const u8, blck_size: u32, type_size: u32, is_quantized: bool };

pub fn traits(t: GGMLType) TypeTraits {
    return switch (t) {
        .GGML_TYPE_F32 => .{ .name = "f32", .blck_size = 1, .type_size = @sizeOf(f32), .is_quantized = false },
        .GGML_TYPE_Q4_K => .{ .name = "q4_K", .blck_size = QK_K, .type_size = @sizeOf(BlockQ4K), .is_quantized = true },
        .GGML_TYPE_Q6_K => .{ .name = "q6_K", .blck_size = QK_K, .type_size = @sizeOf(BlockQ6K), .is_quantized = true },
    };
}

test "traits" {
    try std.testing.expectEqual(144, traits(.GGML_TYPE_Q4_K).type_size);
    try std.testing.expectEqual(210, traits(.GGML_TYPE_Q6_K).type_size);
}
