#pragma once

// A minimal JSON value, parser and writer scoped to the gsstudio-media-helper
// request/response protocol (see README.md). Not a general-purpose JSON
// library: no comments, no trailing commas, doubles only for numbers. The
// protocol is produced by gsstudio itself, not untrusted external input, so this
// narrower implementation is preferred over vendoring a full JSON library.

#include <cctype>
#include <cstdio>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gsstudio::json {

class Value;
using Array = std::vector<Value>;
using Object = std::map<std::string, Value>;

enum class Type { Null, Bool, Number, String, Array, Object };

class Value {
 public:
  Value() : type_(Type::Null) {}
  Value(std::nullptr_t) : type_(Type::Null) {}
  Value(bool value) : type_(Type::Bool), bool_(value) {}
  Value(double value) : type_(Type::Number), number_(value) {}
  Value(int64_t value) : type_(Type::Number), number_(static_cast<double>(value)) {}
  Value(std::string value) : type_(Type::String), string_(std::move(value)) {}
  Value(const char* value) : type_(Type::String), string_(value) {}
  Value(Array value) : type_(Type::Array), array_(std::move(value)) {}
  Value(Object value) : type_(Type::Object), object_(std::move(value)) {}

  Type type() const { return type_; }
  bool is_null() const { return type_ == Type::Null; }
  bool is_object() const { return type_ == Type::Object; }
  bool is_array() const { return type_ == Type::Array; }
  bool is_string() const { return type_ == Type::String; }

  bool as_bool() const { return bool_; }
  double as_number() const { return number_; }
  int64_t as_int() const { return static_cast<int64_t>(number_); }
  const std::string& as_string() const { return string_; }
  const Array& as_array() const { return array_; }
  const Object& as_object() const { return object_; }

  // Object field access. Throws if this is not an object or the key is
  // absent; callers that need an optional field should check `contains()`.
  bool contains(const std::string& key) const {
    return type_ == Type::Object && object_.count(key) > 0;
  }
  const Value& at(const std::string& key) const {
    if (type_ != Type::Object) {
      throw std::runtime_error("json: not an object when reading key '" + key + "'");
    }
    auto found = object_.find(key);
    if (found == object_.end()) {
      throw std::runtime_error("json: missing required key '" + key + "'");
    }
    return found->second;
  }
  const Value& at_or(const std::string& key, const Value& fallback) const {
    if (type_ == Type::Object) {
      auto found = object_.find(key);
      if (found != object_.end()) {
        return found->second;
      }
    }
    return fallback;
  }

 private:
  Type type_;
  bool bool_ = false;
  double number_ = 0.0;
  std::string string_;
  Array array_;
  Object object_;
};

// ---- Parsing ----

class ParseError : public std::runtime_error {
 public:
  explicit ParseError(const std::string& message) : std::runtime_error(message) {}
};

namespace detail {

class Parser {
 public:
  explicit Parser(const std::string& text) : text_(text) {}

  Value Parse() {
    SkipWhitespace();
    Value value = ParseValue();
    SkipWhitespace();
    if (pos_ != text_.size()) {
      throw ParseError("json: unexpected trailing content");
    }
    return value;
  }

 private:
  const std::string& text_;
  size_t pos_ = 0;

  char Peek() {
    if (pos_ >= text_.size()) throw ParseError("json: unexpected end of input");
    return text_[pos_];
  }
  char Next() {
    char c = Peek();
    ++pos_;
    return c;
  }
  void Expect(char expected) {
    char c = Next();
    if (c != expected) {
      throw ParseError(std::string("json: expected '") + expected + "' got '" + c + "'");
    }
  }
  void SkipWhitespace() {
    while (pos_ < text_.size()) {
      char c = text_[pos_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        ++pos_;
      } else {
        break;
      }
    }
  }
  bool Consume(const std::string& literal) {
    if (text_.compare(pos_, literal.size(), literal) == 0) {
      pos_ += literal.size();
      return true;
    }
    return false;
  }

  Value ParseValue() {
    SkipWhitespace();
    char c = Peek();
    if (c == '{') return ParseObject();
    if (c == '[') return ParseArray();
    if (c == '"') return Value(ParseString());
    if (c == 't' || c == 'f') return ParseBool();
    if (c == 'n') {
      if (!Consume("null")) throw ParseError("json: invalid literal");
      return Value(nullptr);
    }
    return ParseNumber();
  }

  Value ParseObject() {
    Expect('{');
    Object object;
    SkipWhitespace();
    if (Peek() == '}') {
      Next();
      return Value(std::move(object));
    }
    while (true) {
      SkipWhitespace();
      std::string key = ParseString();
      SkipWhitespace();
      Expect(':');
      Value value = ParseValue();
      object.emplace(std::move(key), std::move(value));
      SkipWhitespace();
      char c = Next();
      if (c == ',') continue;
      if (c == '}') break;
      throw ParseError("json: expected ',' or '}' in object");
    }
    return Value(std::move(object));
  }

  Value ParseArray() {
    Expect('[');
    Array array;
    SkipWhitespace();
    if (Peek() == ']') {
      Next();
      return Value(std::move(array));
    }
    while (true) {
      array.push_back(ParseValue());
      SkipWhitespace();
      char c = Next();
      if (c == ',') continue;
      if (c == ']') break;
      throw ParseError("json: expected ',' or ']' in array");
    }
    return Value(std::move(array));
  }

  std::string ParseString() {
    Expect('"');
    std::string result;
    while (true) {
      char c = Next();
      if (c == '"') break;
      if (c == '\\') {
        char escaped = Next();
        switch (escaped) {
          case '"': result.push_back('"'); break;
          case '\\': result.push_back('\\'); break;
          case '/': result.push_back('/'); break;
          case 'b': result.push_back('\b'); break;
          case 'f': result.push_back('\f'); break;
          case 'n': result.push_back('\n'); break;
          case 'r': result.push_back('\r'); break;
          case 't': result.push_back('\t'); break;
          case 'u': {
            unsigned code = ParseHex4();
            // Encode the code point as UTF-8. Surrogate pairs are not
            // expected in this protocol's payloads (Windows paths, plain
            // ASCII field names) so they are rejected rather than combined.
            if (code >= 0xD800 && code <= 0xDFFF) {
              throw ParseError("json: surrogate pairs are not supported");
            }
            AppendUtf8(result, code);
            break;
          }
          default:
            throw ParseError("json: invalid escape sequence");
        }
      } else {
        result.push_back(c);
      }
    }
    return result;
  }

  unsigned ParseHex4() {
    unsigned value = 0;
    for (int i = 0; i < 4; ++i) {
      char c = Next();
      value <<= 4;
      if (c >= '0' && c <= '9') value |= static_cast<unsigned>(c - '0');
      else if (c >= 'a' && c <= 'f') value |= static_cast<unsigned>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') value |= static_cast<unsigned>(c - 'A' + 10);
      else throw ParseError("json: invalid \\u escape");
    }
    return value;
  }

  static void AppendUtf8(std::string& out, unsigned code) {
    if (code <= 0x7F) {
      out.push_back(static_cast<char>(code));
    } else if (code <= 0x7FF) {
      out.push_back(static_cast<char>(0xC0 | (code >> 6)));
      out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xE0 | (code >> 12)));
      out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    }
  }

  Value ParseBool() {
    if (Consume("true")) return Value(true);
    if (Consume("false")) return Value(false);
    throw ParseError("json: invalid literal");
  }

  Value ParseNumber() {
    size_t start = pos_;
    if (pos_ < text_.size() && text_[pos_] == '-') ++pos_;
    while (pos_ < text_.size() && isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    if (pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      while (pos_ < text_.size() && isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
      while (pos_ < text_.size() && isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    }
    if (pos_ == start) throw ParseError("json: invalid number");
    return Value(std::stod(text_.substr(start, pos_ - start)));
  }
};

}  // namespace detail

inline Value Parse(const std::string& text) { return detail::Parser(text).Parse(); }

// ---- Writing ----
// Only what the response protocol needs: objects/arrays/strings/numbers/bools,
// written with escaping suitable for the ASCII-safe field values this helper
// ever emits (version strings, camera model names, absolute Windows paths).

inline void WriteEscapedString(std::string& out, const std::string& value) {
  out.push_back('"');
  for (char c : value) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buffer[8];
          snprintf(buffer, sizeof(buffer), "\\u%04x", c);
          out += buffer;
        } else {
          out.push_back(c);
        }
    }
  }
  out.push_back('"');
}

// A tiny streaming object writer so response-building code stays readable
// without constructing an intermediate Value tree.
class ObjectWriter {
 public:
  explicit ObjectWriter(std::string& out) : out_(out) { out_.push_back('{'); }
  ~ObjectWriter() { out_.push_back('}'); }

  ObjectWriter& Field(const std::string& key, const std::string& value) {
    Separator();
    WriteEscapedString(out_, key);
    out_.push_back(':');
    WriteEscapedString(out_, value);
    return *this;
  }
  ObjectWriter& Field(const std::string& key, const char* value) {
    return Field(key, std::string(value));
  }
  ObjectWriter& Field(const std::string& key, bool value) {
    Separator();
    WriteEscapedString(out_, key);
    out_ += value ? ":true" : ":false";
    return *this;
  }
  ObjectWriter& Field(const std::string& key, int64_t value) {
    Separator();
    WriteEscapedString(out_, key);
    out_ += ":" + std::to_string(value);
    return *this;
  }
  ObjectWriter& Field(const std::string& key, double value) {
    Separator();
    WriteEscapedString(out_, key);
    out_ += ":" + std::to_string(value);
    return *this;
  }
  // Emits `"key":[...]` where `write_items` appends comma-separated raw JSON.
  template <typename WriteItemsFn>
  ObjectWriter& RawArray(const std::string& key, WriteItemsFn write_items) {
    Separator();
    WriteEscapedString(out_, key);
    out_.push_back(':');
    out_.push_back('[');
    write_items();
    out_.push_back(']');
    return *this;
  }
  // Emits `"key":{...}` via a nested ObjectWriter passed to `write_fields`.
  template <typename WriteFieldsFn>
  ObjectWriter& RawObject(const std::string& key, WriteFieldsFn write_fields) {
    Separator();
    WriteEscapedString(out_, key);
    out_.push_back(':');
    ObjectWriter nested(out_);
    write_fields(nested);
    return *this;
  }

 private:
  void Separator() {
    if (started_) out_.push_back(',');
    started_ = true;
  }
  std::string& out_;
  bool started_ = false;
};

}  // namespace gsstudio::json
