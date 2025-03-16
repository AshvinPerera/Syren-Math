#pragma once

#include <limits>
#include <new>


namespace SyrenUtility {
	/** Aligned allocator for use with STL containers.
	 *
	 * @details
	 * This allocator is used to ensure proper alignment of memory allocated by STL containers.
	 * It is particularly useful when working with SIMD instructions that require aligned memory.
	 *
	 * @tparam T The type of elements to allocate.
	 * @tparam Alignment The alignment in bytes for the allocated memory.
	 */
	template<typename T, std::size_t Alignment>
	class AlignedAllocator {
	private:
		/** Ensures that the specified alignment is valid.
		 *
		 * @details
		 * The alignment must be at least as large as `alignof(T)` and must be a power of 2.
		 * If these conditions are not met, a static assertion will trigger a compilation error.
		 */
		static_assert(
			Alignment >= alignof(T),
			"The minimum alignment should be greater than the size of the stored type."
			);
		static_assert(
			(Alignment & (Alignment - 1)) == 0,
			"The alignment should be a power of 2."
			);
	public:
		using value_type = T; /*!< The type of elements to allocate. */
		static std::align_val_t constexpr ALIGNMENT{ Alignment }; /*!< The alignment in bytes for the allocated memory. */

		/** Provides a rebind mechanism for STL container compatibility.
		 * 
		 *
		 * @details
		 * This allows an `AlignedAllocator<U, Alignment>` to be used when an allocator
		 * needs to allocate different types within a container.
		 *
		 * @tparam U The new type for which an allocator is being created.
		 */
		template<class U>
		struct rebind
		{
			using other = AlignedAllocator<U, Alignment>;
		};
	public:
		constexpr AlignedAllocator() noexcept = default;
		constexpr AlignedAllocator(const AlignedAllocator&) noexcept = default;
		template<typename U>
		constexpr AlignedAllocator(AlignedAllocator<U, Alignment> const&) noexcept { }
		constexpr ~AlignedAllocator() noexcept { }

		/** Returns the address of a given reference (const reference).
		 *
		 * @param r The reference (const reference) whose address is needed.
		 * @return A pointer to `r`.
		 */
		constexpr T* address(T& r) { return &r; }
		constexpr const T* address(const T& r) const { return &r; }

		/** Allocates memory for `nElementsToAllocate` elements of type `T`.
		 *
		 * @details
		 * The allocated memory will be aligned to the specified alignment.
		 * If the requested allocation exceeds the maximum allowable size, an exception is thrown.
		 *
		 * @param nElementsToAllocate The number of elements to allocate.
		 * @return A pointer to the allocated and aligned memory.
		 * @throws std::bad_array_new_length If the requested size is too large.
		 * @throws std::bad_alloc If allocation fails.
		 */
		[[nodiscard]] 
		T* allocate(std::size_t nElementsToAllocate) {
			if (nElementsToAllocate > (std::numeric_limits<std::size_t>::max)() / sizeof(T)) {
				throw std::bad_array_new_length();
			}

			auto const nBytesToAllocate = nElementsToAllocate * sizeof(T);
			return reinterpret_cast<T*>(::operator new[](nBytesToAllocate, ALIGNMENT));
		}

		/** Deallocates memory for `nBytesAllocated` bytes of memory.
				 *
				 * @details
				 * The memory must have been previously allocated by this allocator.
				 * If the pointer is null, no action is taken.
				 *
				 * @param allocatedPointer The pointer to the memory to deallocate.
				 * @param nBytesAllocated The number of bytes allocated.
				 */
		void deallocate(T* allocatedPointer, [[maybe_unused]] std::size_t  nBytesAllocated) {
			if (allocatedPointer) {
				::operator delete[](allocatedPointer, ALIGNMENT);
			}			
		}

		/** Compares two instances of `AlignedAllocator` for equality.
		 *
		 * @details
		 * Allocators of the same type and alignment are considered equal.
		 *
		 * @tparam U The type used in the allocator being compared.
		 * @return `true`, since all instances of `AlignedAllocator` with the same alignment are equivalent.
		 */
		template <class U>
		bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept { return true; }

		/** Compares two instances of `AlignedAllocator` for inequality.
		 *
		 * @details
		 * Allocators of the same type and alignment are considered equal.
		 *
		 * @tparam U The type used in the allocator being compared.
		 * @return `false`, since all instances of `AlignedAllocator` with the same alignment are equivalent.
		 */
		template <class U>
		bool operator!=(const AlignedAllocator<U, Alignment>&) const noexcept { return false; }
	};
}