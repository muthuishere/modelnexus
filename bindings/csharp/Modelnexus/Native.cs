using System.Runtime.InteropServices;

namespace Modelnexus;

/// <summary>Thrown when the native bridge cannot be located or loaded.</summary>
/// <remarks>
/// Raised on first use rather than at type-load, so that merely referencing this
/// assembly in an environment without the native library is not itself an error.
/// </remarks>
public sealed class NativeLibraryNotFoundException : Exception
{
    internal NativeLibraryNotFoundException(string message) : base(message) { }
}

/// <summary>P/Invoke declarations mirroring core/include/llamabridge.h.</summary>
internal static partial class Native
{
    private const string Lib = "llamabridge";

    // void (*)(const char* text, void* user_data)
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate void StringCallback(IntPtr text, IntPtr userData);

    // int (*)(const char* token_piece, void* user_data) -- non-zero STOPS generation.
    // Separate from StringCallback because of that return value: a void trampoline
    // here would make every stream run to completion no matter what the consumer
    // wants, and the mismatched signature is undefined behaviour besides.
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate int TokenCallback(IntPtr tokenPiece, IntPtr userData);

    static Native()
    {
        // A custom resolver rather than relying on the default probing paths: the
        // library lives in core/dist/<platform>/ during development and beside the
        // assembly once packaged, and neither is somewhere the runtime looks.
        NativeLibrary.SetDllImportResolver(typeof(Native).Assembly, (name, asm, path) =>
        {
            if (name != Lib) return IntPtr.Zero;
            foreach (var candidate in Candidates())
            {
                if (File.Exists(candidate) && NativeLibrary.TryLoad(candidate, out var handle))
                    return handle;
            }
            throw new NativeLibraryNotFoundException(
                "could not locate the modelnexus native bridge.\nLooked in:\n  " +
                string.Join("\n  ", Candidates()) +
                "\nBuild it with core/build.sh, or set MODELNEXUS_LIB to the library or its directory.");
        });
    }

    internal static string PlatformKey()
    {
        var os = RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ? "darwin"
               : RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "windows"
               : "linux";
        var arch = RuntimeInformation.ProcessArchitecture switch
        {
            Architecture.X64 => "x86_64",
            Architecture.Arm64 => "aarch64",
            _ => RuntimeInformation.ProcessArchitecture.ToString().ToLowerInvariant()
        };
        return $"{os}-{arch}";
    }

    private static string LibFileName() =>
        RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ? "libllamabridge.dylib"
      : RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "llamabridge.dll"
      : "libllamabridge.so";

    /// <summary>Search order, most explicit first.</summary>
    private static List<string> Candidates()
    {
        var name = LibFileName();
        var found = new List<string>();

        var env = Environment.GetEnvironmentVariable("MODELNEXUS_LIB");
        if (!string.IsNullOrEmpty(env))
            found.Add(Directory.Exists(env) ? Path.Combine(env, name) : env);

        var baseDir = AppContext.BaseDirectory;
        found.Add(Path.Combine(baseDir, name));
        found.Add(Path.Combine(baseDir, "runtimes", PlatformKey(), "native", name));

        // Walk up to the repo's own build output, for working in the tree.
        var dir = new DirectoryInfo(baseDir);
        for (var i = 0; i < 8 && dir is not null; i++, dir = dir.Parent)
            found.Add(Path.Combine(dir.FullName, "core", "dist", PlatformKey(), name));

        return found;
    }

    // Strings the caller owns come back as IntPtr, never as string: the default
    // marshaller would copy and discard the pointer, making llb_string_free
    // impossible to call and leaking every result.

    [DllImport(Lib, EntryPoint = "llb_version", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr Version();

    [DllImport(Lib, EntryPoint = "llb_model_info", CallingConvention = CallingConvention.Cdecl,
               CharSet = CharSet.Ansi, BestFitMapping = false)]
    internal static extern IntPtr ModelInfo([MarshalAs(UnmanagedType.LPUTF8Str)] string ggufPath);

    // configJson is nullable on purpose: a null string marshals to a NULL pointer,
    // which is how the core is told "defaults" -- distinct from "{}".
    [DllImport(Lib, EntryPoint = "llb_chat_create", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr ChatCreate([MarshalAs(UnmanagedType.LPUTF8Str)] string ggufPath,
                                             [MarshalAs(UnmanagedType.LPUTF8Str)] string? configJson,
                                             StringCallback? eventCb, IntPtr userData);

    [DllImport(Lib, EntryPoint = "llb_chat_infer", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr ChatInfer(IntPtr chat, [MarshalAs(UnmanagedType.LPUTF8Str)] string requestJson);

    [DllImport(Lib, EntryPoint = "llb_chat_infer_stream", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr ChatInferStream(IntPtr chat,
                                                  [MarshalAs(UnmanagedType.LPUTF8Str)] string requestJson,
                                                  TokenCallback? tokenCb, IntPtr userData);

    [DllImport(Lib, EntryPoint = "llb_count_tokens", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr CountTokens(IntPtr chat, [MarshalAs(UnmanagedType.LPUTF8Str)] string requestJson);

    [DllImport(Lib, EntryPoint = "llb_chat_cache", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr ChatCache(IntPtr chat, [MarshalAs(UnmanagedType.LPUTF8Str)] string requestJson);

    [DllImport(Lib, EntryPoint = "llb_chat_lora", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr ChatLora(IntPtr chat, [MarshalAs(UnmanagedType.LPUTF8Str)] string requestJson);

    [DllImport(Lib, EntryPoint = "llb_chat_destroy", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void ChatDestroy(IntPtr chat);

    [DllImport(Lib, EntryPoint = "llb_embed_create", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr EmbedCreate([MarshalAs(UnmanagedType.LPUTF8Str)] string ggufPath,
                                              [MarshalAs(UnmanagedType.LPUTF8Str)] string configJson,
                                              StringCallback? eventCb, IntPtr userData);

    [DllImport(Lib, EntryPoint = "llb_embed", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr Embed(IntPtr embed, [MarshalAs(UnmanagedType.LPUTF8Str)] string requestJson);

    [DllImport(Lib, EntryPoint = "llb_rerank", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr Rerank(IntPtr embed, [MarshalAs(UnmanagedType.LPUTF8Str)] string requestJson);

    [DllImport(Lib, EntryPoint = "llb_embed_destroy", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void EmbedDestroy(IntPtr embed);

    [DllImport(Lib, EntryPoint = "llb_string_free", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void StringFree(IntPtr s);

    [DllImport(Lib, EntryPoint = "llb_set_log_level", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void SetLogLevel(int level);

    // void (*)(int level, const char* text, void* user_data)
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate void LogCallback(int level, IntPtr text, IntPtr userData);

    [DllImport(Lib, EntryPoint = "llb_set_log_callback", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void SetLogCallback(LogCallback? cb, IntPtr userData);

    /// <summary>Copy a core-owned string and free the original.</summary>
    /// <remarks>
    /// Copy and free together, in one place, so the free cannot be forgotten at one
    /// of the several call sites that need it.
    /// </remarks>
    internal static string TakeString(IntPtr ptr)
    {
        if (ptr == IntPtr.Zero) return string.Empty;
        try { return Marshal.PtrToStringUTF8(ptr) ?? string.Empty; }
        finally { StringFree(ptr); }
    }

    /// <summary>Read a string the core still owns.</summary>
    internal static string BorrowString(IntPtr ptr) =>
        ptr == IntPtr.Zero ? string.Empty : Marshal.PtrToStringUTF8(ptr) ?? string.Empty;
}
