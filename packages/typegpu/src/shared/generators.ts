/**
 * Yields values in the sequence 0,1,2..∞ except for the ones in the `excluded` set.
 */
export function* naturalsExcept(
  excluded: Set<number>,
): Generator<number, number, unknown> {
  let next = 0;

  while (true) {
    if (!excluded.has(next)) {
      yield next;
    }

    next++;
  }
}

/**
 * Yields combinations of letters from `components` of given `length`.
 *
 * @example
 * vectorComponentCombinations('xyz', 2)  // xx, xy, xz, yx, yy ...
 */
export function* vectorComponentCombinations(
  components: string,
  length: number,
): Generator<string, undefined, undefined> {
  if (length > 1) {
    for (const str of vectorComponentCombinations(components, length - 1)) {
      for (const component of components) {
        yield str + component;
      }
    }
  } else {
    yield* components;
  }
}
