#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#define MMAP_THRESHOLD_BYTES (64 * 1024) // 64 KiB
#define INITIAL_OFFSETS_CAP 32

#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API
#endif

typedef struct {
    int32_t success;
    char* target_file;
    int64_t start_line;
    int64_t end_line;
    int64_t lines_delta;
    int64_t occurrences;
    char* error;
} PatchResult;

static inline void free_patch_result(PatchResult* res) {
    if (res->target_file) free(res->target_file);
    if (res->error) free(res->error);
}

static void normalize_path_slashes(char* path) {
    for (char* p = path; *p; ++p) {
        if (*p == '\\') *p = '/';
    }
}

static wchar_t* utf8_to_wide(const char* utf8) {
    if (!utf8) return NULL;
    int len = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, NULL, 0);
    if (len <= 0) return NULL;
    wchar_t* wide = (wchar_t*)malloc(len * sizeof(wchar_t));
    if (!wide) return NULL;
    MultiByteToWideChar(CP_UTF8, 0, utf8, -1, wide, len);
    return wide;
}

static int64_t count_newlines_avx2(const uint8_t* data, size_t len) {
    int64_t count = 0;
    size_t i = 0;

    if (len >= 32) {
        __m256i target = _mm256_set1_epi8('\n');
        size_t limit = len - 31;
        for (; i < limit; i += 32) {
            __m256i chunk = _mm256_loadu_si256((const __m256i*)(data + i));
            __m256i cmp = _mm256_cmpeq_epi8(chunk, target);
            uint32_t mask = (uint32_t)_mm256_movemask_epi8(cmp);
            count += __builtin_popcount(mask);
        }
    }

    for (; i < len; ++i) {
        if (data[i] == '\n') {
            count++;
        }
    }
    return count;
}

static const char* detect_dominant_newline(const uint8_t* data, size_t len) {
    size_t probe_len = len > 65536 ? 65536 : len;
    int64_t crlf_count = 0;
    int64_t lf_count = 0;

    for (size_t i = 0; i < probe_len; ++i) {
        if (data[i] == '\n') {
            if (i > 0 && data[i - 1] == '\r') {
                crlf_count++;
            } else {
                lf_count++;
            }
        }
    }
    return (crlf_count > lf_count) ? "\r\n" : "\n";
}

static uint8_t* normalize_newlines_to_bytes(const char* input, size_t in_len, const char* target_nl, size_t* out_len) {
    size_t target_nl_len = strlen(target_nl);
    size_t max_out = in_len * 2 + 16;
    uint8_t* out = (uint8_t*)malloc(max_out);
    if (!out) return NULL;

    size_t j = 0;
    for (size_t i = 0; i < in_len; ++i) {
        if (input[i] == '\r') {
            if (i + 1 < in_len && input[i + 1] == '\n') {
                i++;
            }
            memcpy(out + j, target_nl, target_nl_len);
            j += target_nl_len;
        } else if (input[i] == '\n') {
            memcpy(out + j, target_nl, target_nl_len);
            j += target_nl_len;
        } else {
            out[j++] = (uint8_t)input[i];
        }
    }
    out[j] = '\0';
    *out_len = j;
    return out;
}

typedef struct {
    size_t* offsets;
    size_t count;
    size_t capacity;
    bool ambiguity_error;
    int total_occurrences;
} SearchMatches;

static SearchMatches find_matches_avx2(
    const uint8_t* haystack,
    size_t haystack_len,
    const uint8_t* needle,
    size_t needle_len,
    bool allow_multiple
) {
    SearchMatches res = {0};
    res.capacity = INITIAL_OFFSETS_CAP;
    res.offsets = (size_t*)malloc(res.capacity * sizeof(size_t));
    if (!res.offsets) return res;

    if (needle_len == 0 || needle_len > haystack_len) {
        return res;
    }

    if (needle_len == 1) {
        uint8_t n0 = needle[0];
        for (size_t i = 0; i < haystack_len; ++i) {
            if (haystack[i] == n0) {
                if (!allow_multiple && res.count >= 1) {
                    res.ambiguity_error = true;
                    res.total_occurrences = 2;
                    return res;
                }
                if (res.count >= res.capacity) {
                    res.capacity *= 2;
                    res.offsets = (size_t*)realloc(res.offsets, res.capacity * sizeof(size_t));
                }
                res.offsets[res.count++] = i;
            }
        }
        return res;
    }

    uint8_t first_byte = needle[0];
    uint8_t last_byte = needle[needle_len - 1];
    size_t inner_len = needle_len - 2;

    size_t max_i = haystack_len - needle_len;
    size_t i = 0;

    __m256i target_first = _mm256_set1_epi8((char)first_byte);
    __m256i target_last = _mm256_set1_epi8((char)last_byte);

    while (i <= max_i) {
        if (i + 31 <= max_i) {
            __m256i chunk_first = _mm256_loadu_si256((const __m256i*)(haystack + i));
            __m256i chunk_last  = _mm256_loadu_si256((const __m256i*)(haystack + i + needle_len - 1));

            __m256i cmp_first = _mm256_cmpeq_epi8(chunk_first, target_first);
            __m256i cmp_last  = _mm256_cmpeq_epi8(chunk_last, target_last);
            uint32_t mask = (uint32_t)_mm256_movemask_epi8(_mm256_and_si256(cmp_first, cmp_last));

            while (mask != 0) {
                uint32_t bit = __builtin_ctz(mask);
                size_t candidate = i + bit;

                bool match = true;
                if (inner_len > 0) {
                    match = (memcmp(haystack + candidate + 1, needle + 1, inner_len) == 0);
                }

                if (match) {
                    if (!allow_multiple) {
                        if (res.count >= 1) {
                            res.ambiguity_error = true;
                            int total = 2;
                            size_t cur = candidate + needle_len;
                            while (cur <= max_i) {
                                if (haystack[cur] == first_byte &&
                                    haystack[cur + needle_len - 1] == last_byte &&
                                    (inner_len == 0 || memcmp(haystack + cur + 1, needle + 1, inner_len) == 0)) {
                                    total++;
                                    cur += needle_len;
                                } else {
                                    cur++;
                                }
                            }
                            res.total_occurrences = total;
                            return res;
                        }
                        res.offsets[res.count++] = candidate;
                    } else {
                        if (res.count >= res.capacity) {
                            res.capacity *= 2;
                            res.offsets = (size_t*)realloc(res.offsets, res.capacity * sizeof(size_t));
                        }
                        res.offsets[res.count++] = candidate;
                        i = candidate + needle_len;
                        goto next_outer;
                    }
                }

                mask &= (mask - 1);
            }
            i += 32;
        } else {
            if (haystack[i] == first_byte && haystack[i + needle_len - 1] == last_byte) {
                if (inner_len == 0 || memcmp(haystack + i + 1, needle + 1, inner_len) == 0) {
                    if (!allow_multiple) {
                        if (res.count >= 1) {
                            res.ambiguity_error = true;
                            res.total_occurrences = 2;
                            return res;
                        }
                        res.offsets[res.count++] = i;
                    } else {
                        if (res.count >= res.capacity) {
                            res.capacity *= 2;
                            res.offsets = (size_t*)realloc(res.offsets, res.capacity * sizeof(size_t));
                        }
                        res.offsets[res.count++] = i;
                        i += needle_len;
                        continue;
                    }
                }
            }
            i++;
        }
    next_outer:;
    }

    return res;
}

static bool atomic_replace_win32(const wchar_t* temp_path, const wchar_t* target_path, char** err_out) {
    int max_retries = 5;
    DWORD delay_ms = 5;

    for (int attempt = 0; attempt < max_retries; ++attempt) {
        if (ReplaceFileW(target_path, temp_path, NULL, REPLACEFILE_IGNORE_MERGE_ERRORS, NULL, NULL)) {
            return true;
        }

        DWORD err = GetLastError();
        if (MoveFileExW(temp_path, target_path, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            return true;
        }

        err = GetLastError();
        if (err == ERROR_ACCESS_DENIED || err == ERROR_SHARING_VIOLATION) {
            if (attempt == max_retries - 1) {
                char msg[256];
                snprintf(msg, sizeof(msg), "Atomic rename failed after retries (Win32 error %lu)", err);
                *err_out = _strdup(msg);
                return false;
            }
            Sleep(delay_ms);
            delay_ms *= 2;
        } else {
            char msg[256];
            snprintf(msg, sizeof(msg), "Atomic rename failed (Win32 error %lu)", err);
            *err_out = _strdup(msg);
            return false;
        }
    }
    return false;
}

EXPORT_API PatchResult apply_block_patch_c(
    const char* target_file,
    const char* search_block,
    const char* replace_block,
    bool allow_multiple
) {
    PatchResult res = {0};
    res.target_file = _strdup(target_file ? target_file : "");
    normalize_path_slashes(res.target_file);

    if (!target_file || strlen(target_file) == 0) {
        res.error = _strdup("Target file path cannot be empty.");
        return res;
    }

    if (!search_block || strlen(search_block) == 0) {
        res.error = _strdup("Search block cannot be empty.");
        return res;
    }

    wchar_t* wide_target = utf8_to_wide(target_file);
    if (!wide_target) {
        res.error = _strdup("Failed to convert target file path to wide string.");
        return res;
    }

    HANDLE hFile = CreateFileW(
        wide_target,
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL,
        OPEN_EXISTING,
        FILE_FLAG_SEQUENTIAL_SCAN,
        NULL
    );

    if (hFile == INVALID_HANDLE_VALUE) {
        DWORD err = GetLastError();
        char msg[512];
        snprintf(msg, sizeof(msg), "File not found or cannot be opened: %s (error %lu)", target_file, err);
        res.error = _strdup(msg);
        free(wide_target);
        return res;
    }

    LARGE_INTEGER file_size_li;
    if (!GetFileSizeEx(hFile, &file_size_li)) {
        res.error = _strdup("Failed to retrieve target file size.");
        CloseHandle(hFile);
        free(wide_target);
        return res;
    }

    int64_t file_size = file_size_li.QuadPart;
    if (file_size == 0) {
        res.error = _strdup("Target file is empty.");
        CloseHandle(hFile);
        free(wide_target);
        return res;
    }

    uint8_t* raw_content = NULL;
    HANDLE hMap = NULL;
    bool is_mmap = (file_size >= MMAP_THRESHOLD_BYTES);

    if (is_mmap) {
        hMap = CreateFileMappingW(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!hMap) {
            res.error = _strdup("Failed to create file mapping.");
            CloseHandle(hFile);
            free(wide_target);
            return res;
        }
        raw_content = (uint8_t*)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (!raw_content) {
            res.error = _strdup("Failed to map view of file.");
            CloseHandle(hMap);
            CloseHandle(hFile);
            free(wide_target);
            return res;
        }
    } else {
        raw_content = (uint8_t*)malloc((size_t)file_size);
        if (!raw_content) {
            res.error = _strdup("Out of memory allocating read buffer.");
            CloseHandle(hFile);
            free(wide_target);
            return res;
        }
        DWORD bytes_read = 0;
        if (!ReadFile(hFile, raw_content, (DWORD)file_size, &bytes_read, NULL) || bytes_read != (DWORD)file_size) {
            res.error = _strdup("Failed reading target file bytes.");
            free(raw_content);
            CloseHandle(hFile);
            free(wide_target);
            return res;
        }
    }

    const char* dominant_nl = detect_dominant_newline(raw_content, (size_t)file_size);

    size_t needle_len = 0;
    uint8_t* needle_bytes = normalize_newlines_to_bytes(
        search_block, strlen(search_block), dominant_nl, &needle_len
    );

    SearchMatches matches = find_matches_avx2(
        raw_content, (size_t)file_size, needle_bytes, needle_len, allow_multiple
    );

    const char* active_nl = dominant_nl;
    if (matches.count == 0 && !matches.ambiguity_error) {
        const char* alt_nl = (strcmp(dominant_nl, "\r\n") == 0) ? "\n" : "\r\n";
        size_t alt_needle_len = 0;
        uint8_t* alt_needle_bytes = normalize_newlines_to_bytes(
            search_block, strlen(search_block), alt_nl, &alt_needle_len
        );

        SearchMatches alt_matches = find_matches_avx2(
            raw_content, (size_t)file_size, alt_needle_bytes, alt_needle_len, allow_multiple
        );

        if (alt_matches.count > 0 || alt_matches.ambiguity_error) {
            free(matches.offsets);
            matches = alt_matches;
            free(needle_bytes);
            needle_bytes = alt_needle_bytes;
            needle_len = alt_needle_len;
            active_nl = alt_nl;
        } else {
            free(alt_needle_bytes);
            free(alt_matches.offsets);
        }
    }

    if (matches.ambiguity_error) {
        char msg[256];
        snprintf(msg, sizeof(msg),
            "Ambiguity error: found %d occurrences of search block. Provide more surrounding context or set allow_multiple=true.",
            matches.total_occurrences);
        res.error = _strdup(msg);
        goto cleanup;
    }

    if (matches.count == 0) {
        res.error = _strdup("Search block not found in target file.");
        goto cleanup;
    }

    size_t first_offset = matches.offsets[0];
    res.start_line = count_newlines_avx2(raw_content, first_offset) + 1;
    int64_t lines_in_search = count_newlines_avx2(needle_bytes, needle_len);
    res.end_line = res.start_line + lines_in_search;

    size_t replace_len = 0;
    const char* rep_str = replace_block ? replace_block : "";
    uint8_t* replace_bytes = normalize_newlines_to_bytes(
        rep_str, strlen(rep_str), active_nl, &replace_len
    );
    int64_t lines_in_replace = count_newlines_avx2(replace_bytes, replace_len);
    res.lines_delta = (lines_in_replace - lines_in_search) * (int64_t)matches.count;
    res.occurrences = (int64_t)matches.count;

    LARGE_INTEGER perf_counter;
    QueryPerformanceCounter(&perf_counter);
    DWORD pid = GetCurrentProcessId();

    wchar_t temp_path[MAX_PATH + 64];
    swprintf(temp_path, MAX_PATH + 64, L"%s.tmp.%lu_%llx", wide_target, pid, (unsigned long long)perf_counter.QuadPart);

    HANDLE hTemp = CreateFileW(
        temp_path,
        GENERIC_WRITE,
        0,
        NULL,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    if (hTemp == INVALID_HANDLE_VALUE) {
        DWORD err = GetLastError();
        char msg[256];
        snprintf(msg, sizeof(msg), "Failed to create temporary patch file (error %lu)", err);
        res.error = _strdup(msg);
        free(replace_bytes);
        goto cleanup;
    }

    size_t last_idx = 0;
    bool write_ok = true;
    DWORD written = 0;

    for (size_t m = 0; m < matches.count; ++m) {
        size_t off = matches.offsets[m];
        size_t chunk_len = off - last_idx;
        if (chunk_len > 0) {
            if (!WriteFile(hTemp, raw_content + last_idx, (DWORD)chunk_len, &written, NULL) || written != chunk_len) {
                write_ok = false;
                break;
            }
        }
        if (replace_len > 0) {
            if (!WriteFile(hTemp, replace_bytes, (DWORD)replace_len, &written, NULL) || written != replace_len) {
                write_ok = false;
                break;
            }
        }
        last_idx = off + needle_len;
    }

    if (write_ok) {
        size_t rem_len = (size_t)file_size - last_idx;
        if (rem_len > 0) {
            if (!WriteFile(hTemp, raw_content + last_idx, (DWORD)rem_len, &written, NULL) || written != rem_len) {
                write_ok = false;
            }
        }
    }

    CloseHandle(hTemp);
    free(replace_bytes);

    if (!write_ok) {
        DeleteFileW(temp_path);
        res.error = _strdup("Failed streaming data to temporary patch file.");
        goto cleanup;
    }

    if (is_mmap) {
        UnmapViewOfFile(raw_content);
        raw_content = NULL;
        CloseHandle(hMap);
        hMap = NULL;
    }
    CloseHandle(hFile);
    hFile = INVALID_HANDLE_VALUE;

    char* swap_err = NULL;
    if (!atomic_replace_win32(temp_path, wide_target, &swap_err)) {
        DeleteFileW(temp_path);
        res.error = swap_err ? swap_err : _strdup("Atomic replace failed.");
        goto cleanup;
    }

    res.success = 1;

cleanup:
    if (needle_bytes) free(needle_bytes);
    if (matches.offsets) free(matches.offsets);
    if (is_mmap) {
        if (raw_content) UnmapViewOfFile(raw_content);
        if (hMap) CloseHandle(hMap);
    } else {
        if (raw_content) free(raw_content);
    }
    if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
    if (wide_target) free(wide_target);

    return res;
}

EXPORT_API void free_patch_result_c(PatchResult* res) {
    if (res) {
        free_patch_result(res);
    }
}

typedef struct {
    char* target_file;
    char* search_block;
    char* replace_block;
    bool allow_multiple;
} JsonPayload;

static void free_json_payload(JsonPayload* p) {
    if (p->target_file) free(p->target_file);
    if (p->search_block) free(p->search_block);
    if (p->replace_block) free(p->replace_block);
}

static const char* skip_whitespace(const char* s) {
    while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') s++;
    return s;
}

static char* parse_json_string(const char** pos) {
    const char* s = *pos;
    if (*s != '"') return NULL;
    s++;

    size_t cap = 256;
    size_t len = 0;
    char* buf = (char*)malloc(cap);
    if (!buf) return NULL;

    while (*s && *s != '"') {
        if (len + 8 >= cap) {
            cap *= 2;
            buf = (char*)realloc(buf, cap);
        }
        if (*s == '\\') {
            s++;
            switch (*s) {
                case '"': buf[len++] = '"'; break;
                case '\\': buf[len++] = '\\'; break;
                case '/': buf[len++] = '/'; break;
                case 'b': buf[len++] = '\b'; break;
                case 'f': buf[len++] = '\f'; break;
                case 'n': buf[len++] = '\n'; break;
                case 'r': buf[len++] = '\r'; break;
                case 't': buf[len++] = '\t'; break;
                case 'u': {
                    s++;
                    unsigned int cp = 0;
                    for (int k = 0; k < 4 && *s; ++k, ++s) {
                        cp <<= 4;
                        if (*s >= '0' && *s <= '9') cp |= (*s - '0');
                        else if (*s >= 'a' && *s <= 'f') cp |= (*s - 'a' + 10);
                        else if (*s >= 'A' && *s <= 'F') cp |= (*s - 'A' + 10);
                    }
                    s--;
                    if (cp <= 0x7F) {
                        buf[len++] = (char)cp;
                    } else if (cp <= 0x7FF) {
                        buf[len++] = (char)(0xC0 | (cp >> 6));
                        buf[len++] = (char)(0x80 | (cp & 0x3F));
                    } else {
                        buf[len++] = (char)(0xE0 | (cp >> 12));
                        buf[len++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                        buf[len++] = (char)(0x80 | (cp & 0x3F));
                    }
                    break;
                }
                default:
                    buf[len++] = *s;
                    break;
            }
        } else {
            buf[len++] = *s;
        }
        s++;
    }
    if (*s == '"') s++;
    buf[len] = '\0';
    *pos = s;
    return buf;
}

static bool parse_json_payload(const char* json_str, JsonPayload* out) {
    memset(out, 0, sizeof(JsonPayload));
    const char* s = skip_whitespace(json_str);
    if (*s != '{') return false;
    s++;

    while (*s) {
        s = skip_whitespace(s);
        if (*s == '}') break;
        if (*s == ',') { s++; continue; }

        char* key = parse_json_string(&s);
        if (!key) break;

        s = skip_whitespace(s);
        if (*s == ':') s++;
        s = skip_whitespace(s);

        if (strcmp(key, "target_file") == 0) {
            out->target_file = parse_json_string(&s);
        } else if (strcmp(key, "search_block") == 0) {
            out->search_block = parse_json_string(&s);
        } else if (strcmp(key, "replace_block") == 0) {
            out->replace_block = parse_json_string(&s);
        } else if (strcmp(key, "allow_multiple") == 0) {
            if (strncmp(s, "true", 4) == 0) {
                out->allow_multiple = true;
                s += 4;
            } else if (strncmp(s, "false", 5) == 0) {
                out->allow_multiple = false;
                s += 5;
            }
        } else {
            if (*s == '"') {
                char* dummy = parse_json_string(&s);
                free(dummy);
            } else {
                while (*s && *s != ',' && *s != '}') s++;
            }
        }
        free(key);
    }
    return true;
}

static char* read_file_to_string(const char* filepath) {
    if (!filepath) return NULL;
    FILE* fp = fopen(filepath, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char* buf = (char*)malloc(sz + 1);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    size_t r = fread(buf, 1, sz, fp);
    fclose(fp);
    buf[r] = '\0';
    return buf;
}

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n\n"
        "High-performance atomic block-patching tool.\n\n"
        "Options:\n"
        "  -h, --help            Show this help message and exit\n"
        "  -f, --file FILE       Path to target file\n"
        "  -s, --search SEARCH   Contiguous text block to find\n"
        "  -r, --replace REPLACE New replacement text block\n"
        "  --allow-multiple      Allow replacing all occurrences (default: false)\n"
        "  -j, --json JSON       JSON payload string\n"
        "  --stdin               Read JSON payload from standard input\n"
        "  --payload-file FILE   Path to JSON file containing parameters\n"
        "  --search-file FILE    Path to file containing search block\n"
        "  --replace-file FILE   Path to file containing replacement block\n"
        "  --clean-tmp           Delete search-file, replace-file, or payload-file on success\n",
        prog
    );
}

#ifndef BUILD_AS_DLL
int main(int argc, char** argv) {
    SetConsoleOutputCP(CP_UTF8);

    char* target_file = NULL;
    char* search_block = NULL;
    char* replace_block = NULL;
    bool allow_multiple = false;
    char* json_arg = NULL;
    bool use_stdin = false;
    char* payload_file = NULL;
    char* search_file = NULL;
    char* replace_file = NULL;
    bool clean_tmp = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if ((strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--file") == 0) && i + 1 < argc) {
            target_file = argv[++i];
        } else if ((strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--search") == 0) && i + 1 < argc) {
            search_block = argv[++i];
        } else if ((strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--replace") == 0) && i + 1 < argc) {
            replace_block = argv[++i];
        } else if (strcmp(argv[i], "--allow-multiple") == 0) {
            allow_multiple = true;
        } else if ((strcmp(argv[i], "-j") == 0 || strcmp(argv[i], "--json") == 0) && i + 1 < argc) {
            json_arg = argv[++i];
        } else if (strcmp(argv[i], "--stdin") == 0) {
            use_stdin = true;
        } else if (strcmp(argv[i], "--payload-file") == 0 && i + 1 < argc) {
            payload_file = argv[++i];
        } else if (strcmp(argv[i], "--search-file") == 0 && i + 1 < argc) {
            search_file = argv[++i];
        } else if (strcmp(argv[i], "--replace-file") == 0 && i + 1 < argc) {
            replace_file = argv[++i];
        } else if (strcmp(argv[i], "--clean-tmp") == 0) {
            clean_tmp = true;
        }
    }

    JsonPayload payload = {0};
    bool free_payload = false;

    if (use_stdin) {
        size_t cap = 65536;
        size_t len = 0;
        char* buf = (char*)malloc(cap);
        if (!buf) return 1;

        size_t r;
        while ((r = fread(buf + len, 1, cap - len - 1, stdin)) > 0) {
            len += r;
            if (len + 1024 >= cap) {
                cap *= 2;
                buf = (char*)realloc(buf, cap);
            }
        }
        buf[len] = '\0';
        if (!parse_json_payload(buf, &payload)) {
            fprintf(stderr, "{\"success\": false, \"error\": \"Invalid JSON on stdin.\"}\n");
            free(buf);
            return 1;
        }
        free(buf);
        free_payload = true;
    } else if (payload_file) {
        FILE* fp = fopen(payload_file, "rb");
        if (!fp) {
            fprintf(stderr, "{\"success\": false, \"error\": \"Failed reading payload file.\"}\n");
            return 1;
        }
        fseek(fp, 0, SEEK_END);
        long sz = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        char* buf = (char*)malloc(sz + 1);
        if (fread(buf, 1, sz, fp) != (size_t)sz) {
            fclose(fp);
            free(buf);
            fprintf(stderr, "{\"success\": false, \"error\": \"Failed reading payload file.\"}\n");
            return 1;
        }
        fclose(fp);
        buf[sz] = '\0';
        if (!parse_json_payload(buf, &payload)) {
            fprintf(stderr, "{\"success\": false, \"error\": \"Invalid JSON in payload file.\"}\n");
            free(buf);
            return 1;
        }
        free(buf);
        free_payload = true;
    } else if (json_arg) {
        if (!parse_json_payload(json_arg, &payload)) {
            fprintf(stderr, "{\"success\": false, \"error\": \"Invalid JSON argument.\"}\n");
            return 1;
        }
        free_payload = true;
    } else if (target_file) {
        payload.target_file = _strdup(target_file);
        if (search_file) {
            payload.search_block = read_file_to_string(search_file);
            if (!payload.search_block) {
                fprintf(stderr, "{\"success\": false, \"error\": \"Failed reading search file.\"}\n");
                free(payload.target_file);
                return 1;
            }
        } else {
            payload.search_block = _strdup(search_block ? search_block : "");
        }

        if (replace_file) {
            payload.replace_block = read_file_to_string(replace_file);
            if (!payload.replace_block) {
                fprintf(stderr, "{\"success\": false, \"error\": \"Failed reading replace file.\"}\n");
                free(payload.target_file);
                free(payload.search_block);
                return 1;
            }
        } else {
            payload.replace_block = _strdup(replace_block ? replace_block : "");
        }
        payload.allow_multiple = allow_multiple;
        free_payload = true;
    } else {
        print_usage(argv[0]);
        return 1;
    }

    if (!payload.target_file || strlen(payload.target_file) == 0) {
        fprintf(stderr, "{\"success\": false, \"error\": \"Missing target_file parameter.\"}\n");
        if (free_payload) free_json_payload(&payload);
        return 1;
    }

    PatchResult res = apply_block_patch_c(
        payload.target_file,
        payload.search_block ? payload.search_block : "",
        payload.replace_block ? payload.replace_block : "",
        payload.allow_multiple
    );

    if (res.success) {
        if (clean_tmp) {
            if (search_file) remove(search_file);
            if (replace_file) remove(replace_file);
            if (payload_file) remove(payload_file);
        }
        printf("{\n"
               "  \"success\": true,\n"
               "  \"target_file\": \"%s\",\n"
               "  \"start_line\": %lld,\n"
               "  \"end_line\": %lld,\n"
               "  \"lines_delta\": %lld,\n"
               "  \"occurrences\": %lld\n"
               "}\n",
               res.target_file,
               (long long)res.start_line,
               (long long)res.end_line,
               (long long)res.lines_delta,
               (long long)res.occurrences);
        free_patch_result(&res);
        if (free_payload) free_json_payload(&payload);
        return 0;
    } else {
        fprintf(stderr, "{\n"
                        "  \"success\": false,\n"
                        "  \"target_file\": \"%s\",\n"
                        "  \"error\": \"%s\"\n"
                        "}\n",
                res.target_file ? res.target_file : "",
                res.error ? res.error : "Unknown error");
        free_patch_result(&res);
        if (free_payload) free_json_payload(&payload);
        return 2;
    }
}
#endif
