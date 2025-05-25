#pragma once

#include <memory>
#include <type_traits>
#include <stdexcept>
#include <string>
#include <typeinfo>

namespace shrew {
    namespace utils {
        template<typename Derived, typename Base>
        std::shared_ptr<Derived> cast_pointer(const std::shared_ptr<Base>& base_ptr, const std::string& context = "") {
            static_assert(std::is_base_of<Base, Derived>::value, "Derived must inherit from Base");

            auto derived_ptr = std::dynamic_pointer_cast<Derived>(base_ptr);
            if (!derived_ptr) {
                throw std::runtime_error(
                    "Failed to cast in " + context +
                    ". Expected type: " + typeid(Derived).name() +
                    ", actual type: " + typeid(*base_ptr).name());
            }
            return derived_ptr;
        }
    }
}