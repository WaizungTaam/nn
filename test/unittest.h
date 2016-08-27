/*
Copyright 2016 Waizung Taam

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

- 2016-08-15

- ======== unittest ========

- namespace unittest
  - namespace internal
    - string-relevant
    - colored terminal printing
  - namespace framework
    - enum TestReportMode
    - class TestState
    - struct Failure
    - class TestFunc
    - class TestCase
    - class TestSuite
    - struct TestSuiteUpdator
  - namespace compare
    - comparison structs
    - comparison and evaluation
  - EXPECT Macros
  - ASSERT Macros
  - class UnitTest
  - TEST, TEST_F Macros
  - RUN_ALL_TESTS

*/

#ifndef UNITTEST_H_
#define UNITTEST_H_

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace unittest {

// ======== namespace internal ========
namespace internal {

// ======== String-Relevant ========
template<typename T>
std::string to_string(const T& obj) {
  std::ostringstream os;
  os << obj;
  return os.str();
}
std::string to_string(bool b) {
  return b ? "true" : "false";
}

std::string make_messsage(const std::string& expected, 
                          const std::string& actual) {
  // return "Excepted " + expected + ", failed at " + actual + ".";
  return "Failed at " + expected + ".";
}
std::string make_messsage(const std::string& expression, bool expected_bool) {
  return "Expected (" + expression + ") == " + to_string(expected_bool) +
    ", but actually (" + expression + ") == " + to_string(!expected_bool) + 
    ".";
}

template <typename LhsT, typename RhsT, typename OpT>
std::string make_expression(const LhsT& lhs, const RhsT& rhs, OpT op) {
  return to_string(lhs) + " " + op.name() + " " + to_string(rhs);
}

// ======== Colored Terminal Printing ========
#define UNITTEST_RESET_     0
#define UNITTEST_BRIGHT_    1
#define UNITTEST_DIM_       2
#define UNITTEST_UNDERLINE_ 4
#define UNITTEST_BLINK_     5
#define UNITTEST_REVERSE_   7
#define UNITTEST_HIDDEN_    8

#define UNITTEST_BLACK_     0
#define UNITTEST_RED_       1
#define UNITTEST_GREEN_     2
#define UNITTEST_YELLOW_    3
#define UNITTEST_BLUE_      4
#define UNITTEST_MAGENTA_   5
#define UNITTEST_CYAN_      6
#define UNITTEST_WHITE_     7

class TerminalColor {
public:
  TerminalColor() :
    attribute_(1), foreground_color_(7), background_color_(0) {}
  TerminalColor(int attr, int fgc, int bgc) :
    attribute_(attr), foreground_color_(fgc), background_color_(bgc) {}
  TerminalColor(const TerminalColor& other) :
    attribute_(other.attribute_), foreground_color_(other.foreground_color_),
    background_color_(other.background_color_) {}
  TerminalColor& operator=(const TerminalColor& other) {
    attribute_ = other.attribute_;
    foreground_color_ = other.foreground_color_;
    background_color_ = other.background_color_;
    return *this;
  }
  ~TerminalColor() {}
  void print() const {
    char control_code[13];
    sprintf(control_code, "%c[%d;%d;%dm", 0x1B, attribute_, 
            foreground_color_ + 30, background_color_ + 40);
    printf("%s", control_code);
  }
private:
  int attribute_;
  int foreground_color_;
  int background_color_;
};

void colored_print(const TerminalColor& color, const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  color.print();
  printf("%s", fmt);
  TerminalColor().print();
  va_end(args);
}
static TerminalColor color_passed(
  UNITTEST_BRIGHT_, UNITTEST_GREEN_, UNITTEST_BLACK_);
static TerminalColor color_failed(
  UNITTEST_BRIGHT_, UNITTEST_RED_, UNITTEST_BLACK_);
static TerminalColor color_test_case(
  UNITTEST_BRIGHT_, UNITTEST_CYAN_, UNITTEST_BLACK_);
static TerminalColor color_test_func(
  UNITTEST_BRIGHT_, UNITTEST_BLUE_, UNITTEST_BLACK_);

}  // namespace internal


// ========= namespace framework ========
namespace framework {

class TestFunc;
class TestCase;

enum TestReportMode {
  report_only_when_failed,
  report_each
};

// ======== Current State of Testing ========
class TestState {
public:
  TestState() : test_func_(nullptr), test_case_(nullptr), 
    report_mode_(report_each) {}
  // non-copyable
  TestState(const TestState&) = delete;
  TestState& operator=(const TestState&) = delete;

  static TestState& get_instance() {
    static TestState instance;
    return instance;
  }

  static TestFunc* get_test_func() { return get_instance().test_func_; }
  static TestCase* get_test_case() { return get_instance().test_case_; }
  static TestReportMode get_report_mode() {
    return get_instance().report_mode_;
  }

  static void set_test_func(TestFunc* test_func) { 
    get_instance().test_func_ = test_func;
  }
  static void set_test_case(TestCase* test_case) {
    get_instance().test_case_ = test_case;
  }
  static void set_report_mode(TestReportMode report_mode) {
    get_instance().report_mode_ = report_mode;
  }

private:
  TestFunc* test_func_;
  TestCase* test_case_;
  TestReportMode report_mode_;
};

// ======== Collection of Failure Information ========
struct Failure {
  Failure() {}

  Failure(const std::string& file_name, std::size_t line,
          const std::string& expected, const std::string& actual) :
    file_name(file_name), line(line), 
    message(internal::make_messsage(expected, actual)) {}

  Failure(const std::string& file_name, std::size_t line,
          const std::string& expression, bool expected_bool) :
    file_name(file_name), line(line), 
    message(internal::make_messsage(expression, expected_bool)) {}

  std::string file_name;
  std::size_t line;
  std::string message;
};

// ======== Test of A Single Function ========
class TestFunc {
public:
  typedef void (*func_t)();

  TestFunc(const std::string& name, func_t function) :
    executed_(false), name_(name), function_(function) {}

  std::string name() const { return name_; }
  bool passed() const { return executed_ && failures_.empty(); }
  bool failed() const { return !passed(); }

  void execute() {
    TestState::set_test_func(this);
    function_();
    executed_ = true;
  }
  void add(const Failure& failure) { 
    failures_.push_back(failure); 
  }
  template <typename CharT, typename Traits>
  void report(std::basic_ostream<CharT, Traits>& os) const {
    for (const Failure& failure : failures_) {
      report_one_failure_(os, failure);
    }
  }

private:
  template <typename CharT, typename Traits>
  void report_one_failure_(std::basic_ostream<CharT, Traits>& os,
                           const Failure& failure) const {
    os << "  ";
    internal::colored_print(internal::color_test_func, name_.c_str());
    os << ": " << failure.file_name << "[" << failure.line 
       << "]: " << failure.message << "\n";
  }

  bool executed_;
  std::string name_;
  func_t function_;
  std::vector<Failure> failures_;
};

// ======== Collection of TestFunc(s) ========
class TestCase {
public:
  TestCase() {}
  TestCase(const std::string& name) : executed_(false), name_(name) {}

  std::string name() const { return name_; }
  bool passed() const {
    if (!executed_) {
      return false;
    }
    for (const TestFunc& test_func : test_funcs_) {
      if (test_func.failed()) {
        return false;
      }
    }
    return true;
  }
  bool failed() const { return !passed(); }

  template <typename CharT, typename Traits>
  void execute(std::basic_ostream<CharT, Traits>& os) {
    TestState::set_test_case(this);
    for (TestFunc& test_func : test_funcs_) {
      test_func.execute();
    }
    executed_ = true;
    if (TestState::get_report_mode() == TestReportMode::report_each) {
      report(os);
    }
  }
  void add(const TestFunc& test_func) {
    test_funcs_.push_back(test_func);
  }
  template <typename CharT, typename Traits>
  void report(std::basic_ostream<CharT, Traits>& os) const {
    internal::colored_print(internal::color_test_case, name_.c_str());
    os << ": ";
    if (passed()) {
      internal::colored_print(internal::color_passed, "[ PASSED ]");
    } else {
      internal::colored_print(internal::color_failed, "[ FAILED ]");
      os << "\n";
    }
    for (const TestFunc& test_func : test_funcs_) {
      if (test_func.failed()) {
        test_func.report(os);
      }
    }
    os << "\n";
  }

private:
  bool executed_;
  std::string name_;
  std::vector<TestFunc> test_funcs_;
};

// ======== Collection of All TestCase(s) in a Test Program ========
class TestSuite {
public:
  static TestSuite& get_instance() {
    static TestSuite instance;
    return instance;
  }

  std::size_t total_num() const { return test_cases_.size(); }
  std::size_t passed_num() const {
    std::size_t num = 0;
    for (const TestCase& test_case : test_cases_) {
      if (test_case.passed()) {
        ++num;
      }
    }
    return num;
  }
  std::size_t failed_num() const { return total_num() - passed_num(); }
  bool passed() const { return failed_num() == 0; }
  bool failed() const { return !passed(); }

  template <typename CharT, typename Traits>
  void execute(std::basic_ostream<CharT, Traits>& os) {
    for (TestCase& test_case : test_cases_) {
      test_case.execute(os);
    }
  }
  void add(const std::string& test_case_name, const TestFunc& test_func) {
    bool exist = false; 
    for (TestCase& test_case : test_cases_) {
      if (test_case.name() == test_case_name) {
        test_case.add(test_func);
        exist = true;
      }
    }
    if (!exist) {
      test_cases_.push_back(TestCase(test_case_name));
      test_cases_[test_cases_.size() - 1].add(test_func);
    }
  }
  template <typename CharT, typename Traits>
  void report(std::basic_ostream<CharT, Traits>& os) const {
    os << "\n";
    if (failed()) {
      internal::colored_print(internal::color_failed, "[ FAILED ]");
      os << " " << failed_num() << " OF " << total_num() 
         << " TEST(S) FAILED.\n";
    } else {
      internal::colored_print(internal::color_passed, "[ ALL PASSED ]");
      os << "\n";
    }
  }

private:
  // Only one TestSuite exists in a program
  TestSuite() {}
  std::vector<TestCase> test_cases_;
};

struct TestSuiteUpdator {
  TestSuiteUpdator(const std::string& test_case_name, 
                   const TestFunc& test_func) {
    TestSuite::get_instance().add(test_case_name, test_func);
  }
};

}  // namespace framework


namespace compare {

// ======== Operators ========
struct EQ {
  static std::string name() { return "=="; }
  template <typename LhsT, typename RhsT>
  bool operator()(const LhsT& lhs, const RhsT& rhs) { return lhs == rhs; }
};
struct NE {
  static std::string name() { return "!="; }
  template <typename LhsT, typename RhsT>
  bool operator()(const LhsT& lhs, const RhsT& rhs) { return lhs != rhs; }
};
struct LT {
  static std::string name() { return "<"; }
  template <typename LhsT, typename RhsT>
  bool operator()(const LhsT& lhs, const RhsT& rhs) { return lhs < rhs; }
};
struct LE {
  static std::string name() { return "<="; }
  template <typename LhsT, typename RhsT>
  bool operator()(const LhsT& lhs, const RhsT& rhs) { return lhs <= rhs; }
};
struct GT {
  static std::string name() { return ">"; }
  template <typename LhsT, typename RhsT>
  bool operator()(const LhsT& lhs, const RhsT& rhs) { return lhs > rhs; }
};
struct GE {
  static std::string name() { return ">="; }
  template <typename LhsT, typename RhsT>
  bool operator()(const LhsT& lhs, const RhsT& rhs) { return lhs >= rhs; }
};
struct FEQ {
  static std::string name() { return "=="; }
  template <typename LhsT, typename RhsT>
  bool operator()(const LhsT& lhs, const RhsT& rhs, std::size_t ulp = 1) {
  static_assert(std::is_floating_point<LhsT>::value &&
                std::is_floating_point<RhsT>::value, 
                "FEQ only compares floating point numbers.");
    return std::fabs(lhs - rhs) <= ulp * std::numeric_limits<LhsT>::epsilon() *
                                   std::fabs((lhs + rhs) / 2.0)
        || std::fabs(lhs - rhs) <= std::numeric_limits<LhsT>::min();  
  }
};
struct FNE {
  static std::string name() { return "!="; }
  template <typename LhsT, typename RhsT>
  bool operator()(const LhsT& lhs, const RhsT& rhs, std::size_t ulp = 1) { 
    return !FEQ()(lhs, rhs, ulp);
  }
};

// ======== Evaluation and Comparision =======
bool evaluate(bool expected, bool actual, const char* expression, 
              const char* file, std::size_t line) {
  bool eval_res = (expected == actual);
  if (!eval_res) {
    framework::TestState::get_test_func()->add(
      framework::Failure(file, line, expression, expected));
  }
  return eval_res;
}

template <typename ExpT, typename ActT, typename OpT>
bool compare(const ExpT& expected, const ActT& actual, OpT operation,
             const char* expected_str, const char* actual_str,
             const char* file, std::size_t line) {
  bool compare_res = operation(expected, actual);
  if (!compare_res) {
    framework::TestState::get_test_func()->add(
      framework::Failure(file, line,
      internal::make_expression(expected_str, actual_str, operation),
      internal::make_expression(expected, actual, operation)));
  }
  return compare_res;
}

template <typename ExpT, typename ActT, typename OpT>
bool compare_floating(const ExpT& expected, const ActT& actual, OpT operation,
                      const std::size_t& ulp,
                      const char* expected_str, const char* actual_str,
                      const char* file, std::size_t line) {
  bool compare_res = operation(expected, actual, ulp);
  if (!compare_res) {
    framework::TestState::get_test_func()->add(
      framework::Failure(file, line,
      internal::make_expression(expected_str, actual_str, operation),
      internal::make_expression(expected, actual, operation)));
  }
  return compare_res;
}

}  // namespace compare

// ======== EXPECT ========
#define EXPECT_BOOL(expected, expression) \
  compare::evaluate(expected, expression, #expression, __FILE__, __LINE__)
#define EXPECT_BINARY(lhs, rhs, op) \
  compare::compare(lhs, rhs, op(), #lhs, #rhs, __FILE__, __LINE__)
#define EXPECT_FLOATING(lhs, rhs, op, ulp) \
  compare::compare_floating(lhs, rhs, op(), ulp, #lhs, #rhs, \
  __FILE__, __LINE__)

#define EXPECT_TRUE(expression) EXPECT_BOOL(true, expression)
#define EXPECT_FALSE(expression) EXPECT_BOOL(false, expression)
#define EXPECT_EQ(expected, actual) \
  EXPECT_BINARY(expected, actual, compare::EQ)
#define EXPECT_NE(expected, actual) \
  EXPECT_BINARY(expected, actual, compare::NE)
#define EXPECT_LT(expected, actual) \
  EXPECT_BINARY(expected, actual, compare::LT)
#define EXPECT_LE(expected, actual) \
  EXPECT_BINARY(expected, actual, compare::LE)
#define EXPECT_GT(expected, actual) \
  EXPECT_BINARY(expected, actual, compare::GT)
#define EXPECT_GE(expected, actual) \
  EXPECT_BINARY(expected, actual, compare::GE)
#define EXPECT_FEQ(expected, actual, ulp) \
  EXPECT_FLOATING(expected, actual, compare::FEQ, ulp)
#define EXPECT_FNE(expected, actual, ulp) \
  EXPECT_FLOATING(expected, actual, compare::FNE, ulp)

// ======== ASSERT ========
#define ASSERT_BOOL(expected, expression) \
do { \
  if (!EXPECT_BOOL(expected, expression)) { \
    internal::colored_print(internal::color_failed, "[ FATAL ERROR ]"); \
    printf("\n"); \
    return; \
  } \
} while (0)

#define ASSERT_BINARY(lhs, rhs, op) \
do { \
  if (!EXPECT_BINARY(lhs, rhs, op)) { \
    internal::colored_print(internal::color_failed, "[ FATAL ERROR ]"); \
    printf("\n"); \
    return; \
  } \
} while (0)

#define ASSERT_FLOATING(lhs, rhs, op, ulp) \
do { \
  if (!EXPECT_FLOATING(lhs, rhs, op, ulp)) { \
    internal::colored_print(internal::color_failed, "[ FATAL ERROR ]"); \
    printf("\n"); \
    return; \
  } \
} while (0)

#define ASSERT_TRUE(expression) ASSERT_BOOL(true, expression)
#define ASSERT_FALSE(expression) ASSERT_BOOL(false, expression)
#define ASSERT_EQ(expected, actual) \
  ASSERT_BINARY(expected, actual, compare::EQ)
#define ASSERT_NE(expected, actual) \
  ASSERT_BINARY(expected, actual, compare::NE)
#define ASSERT_LT(expected, actual) \
  ASSERT_BINARY(expected, actual, compare::LT)
#define ASSERT_LE(expected, actual) \
  ASSERT_BINARY(expected, actual, compare::LE)
#define ASSERT_GT(expected, actual) \
  ASSERT_BINARY(expected, actual, compare::GT)
#define ASSERT_GE(expected, actual) \
  ASSERT_BINARY(expected, actual, compare::GE)
#define ASSERT_FEQ(expected, actual, ulp) \
  ASSERT_FLOATING(expected, actual, compare::FEQ, ulp)
#define ASSERT_FNE(expected, actual, ulp) \
  ASSERT_FLOATING(expected, actual, compare::FNE, ulp)


// ======== class UnitTest ========
class UnitTest {
public:
  UnitTest() {}
  virtual ~UnitTest() {}
  void execute() {
    setup();
    test_method();
    teardown();
  }
protected:
  virtual void setup() {}
  virtual void test_method() = 0;
  virtual void teardown() {}
};

// ======== TEST, TEST_F ========
// Macros prefixed with UNITTEST_ and suffixed with _ are not intended to
// be used by users.
namespace internal {

#define UNITTEST_STR_(x) #x
#define UNITTEST_JOIN_(X, Y) UNITTEST_DO_JOIN_( X, Y )
#define UNITTEST_DO_JOIN_( X, Y ) UNITTEST_DO_JOIN2_(X,Y)
#define UNITTEST_DO_JOIN2_( X, Y ) X##_##Y
#define UNITTEST_IDENTIFIER_(test_case, test_func) \
  UNITTEST_JOIN_(test_case, test_func)
#define UNITTEST_INVOKER_(test_case, test_func) \
  UNITTEST_JOIN_(UNITTEST_IDENTIFIER_(test_case, test_func), invoker_)
#define UNITTEST_UPDATOR_(test_case, test_func) \
  static framework::TestSuiteUpdator \
  UNITTEST_JOIN_(UNITTEST_IDENTIFIER_(test_case, test_func), updator_)
#define UNITTEST_CREATE_TEST_FUNC(test_case, test_func) \
  framework::TestFunc(UNITTEST_STR_(test_func), \
  UNITTEST_INVOKER_(test_case, test_func))

#define UNITTEST_TEST_CASE_ALLOC_(test_case, test_func, base) \
struct UNITTEST_IDENTIFIER_(test_case, test_func) : public base { \
  void test_method(); \
}; \
void UNITTEST_INVOKER_(test_case, test_func)() { \
  UNITTEST_IDENTIFIER_(test_case, test_func) test_obj; \
  test_obj.execute(); \
} \
UNITTEST_UPDATOR_(test_case, test_func)(UNITTEST_STR_(test_case), \
  UNITTEST_CREATE_TEST_FUNC(test_case, test_func)); \
void UNITTEST_IDENTIFIER_(test_case, test_func)::test_method()

}  // namespace internal

#define TEST(test_case, test_func) \
  UNITTEST_TEST_CASE_ALLOC_(test_case, test_func, UnitTest)
#define TEST_F(test_fixture, test_func) \
  UNITTEST_TEST_CASE_ALLOC_(test_fixture, test_func, test_fixture)


// ======== RUN_ALL_TESTS ========
bool RUN_ALL_TESTS() {
  framework::TestSuite::get_instance().execute(std::cerr);
  framework::TestSuite::get_instance().report(std::cerr);
  return framework::TestSuite::get_instance().passed();
}

}  // namespace unittest

#endif  // UNITTEST_H_