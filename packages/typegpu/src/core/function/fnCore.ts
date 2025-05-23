import { getAttributesString } from '../../data/attributes.ts';
import {
  type AnyWgslData,
  isWgslData,
  isWgslStruct,
  Void,
} from '../../data/wgslTypes.ts';
import { MissingLinksError } from '../../errors.ts';
import { getName, setName } from '../../name.ts';
import type { ResolutionCtx, Snippet } from '../../types.ts';
import {
  addArgTypesToExternals,
  addReturnTypeToExternals,
  applyExternals,
  type ExternalMap,
  replaceExternalsInWgsl,
} from '../resolve/externals.ts';
import { getPrebuiltAstFor } from './astUtils.ts';
import type { Implementation } from './fnTypes.ts';

export interface TgpuFnShellBase<Args extends unknown[], Return> {
  readonly argTypes: Args;
  readonly returnType: Return;
  readonly isEntry: boolean;
}

export interface FnCore {
  applyExternals(newExternals: ExternalMap): void;
  resolve(ctx: ResolutionCtx, fnAttribute?: string): string;
}

export function createFnCore(
  shell: TgpuFnShellBase<unknown[], unknown>,
  implementation: Implementation,
): FnCore {
  /**
   * External application has to be deferred until resolution because
   * some externals can reference the owner function which has not been
   * initialized yet (like when accessing the Output struct of a vertex
   * entry fn).
   */
  const externalsToApply: ExternalMap[] = [];

  if (typeof implementation === 'string') {
    if (!shell.isEntry) {
      addArgTypesToExternals(
        implementation,
        shell.argTypes,
        (externals) => externalsToApply.push(externals),
      );
      addReturnTypeToExternals(
        implementation,
        shell.returnType,
        (externals) => externalsToApply.push(externals),
      );
    } else {
      if (isWgslStruct(shell.argTypes[0])) {
        externalsToApply.push({ In: shell.argTypes[0] });
      }

      if (isWgslStruct(shell.returnType)) {
        externalsToApply.push({ Out: shell.returnType });
      }
    }
  }

  const core = {
    applyExternals(newExternals: ExternalMap): void {
      externalsToApply.push(newExternals);
    },

    resolve(ctx: ResolutionCtx, fnAttribute = ''): string {
      const externalMap: ExternalMap = {};

      for (const externals of externalsToApply) {
        applyExternals(externalMap, externals);
      }

      const id = ctx.names.makeUnique(getName(this));

      if (typeof implementation === 'string') {
        let header = '';

        if (shell.isEntry) {
          const input = isWgslStruct(shell.argTypes[0]) ? '(in: In)' : '()';

          const attributes = isWgslData(shell.returnType)
            ? getAttributesString(shell.returnType)
            : '';
          const output = shell.returnType !== Void
            ? isWgslStruct(shell.returnType)
              ? '-> Out'
              : `-> ${attributes !== '' ? attributes : '@location(0)'} ${
                ctx.resolve(shell.returnType)
              }`
            : '';
          header = `${input} ${output} `;
        }

        const replacedImpl = replaceExternalsInWgsl(
          ctx,
          externalMap,
          `${header}${implementation.trim()}`,
        );

        ctx.addDeclaration(`${fnAttribute}fn ${id}${replacedImpl}`);
      } else {
        // get data generated by the plugin
        const pluginData = getPrebuiltAstFor(implementation);

        if (pluginData?.externals) {
          const missing = Object.fromEntries(
            Object.entries(pluginData.externals).filter(
              ([name]) => !(name in externalMap),
            ),
          );

          applyExternals(externalMap, missing);
        }
        const ast = pluginData?.ast ?? ctx.transpileFn(String(implementation));

        if (ast.argNames.type === 'destructured-object') {
          applyExternals(
            externalMap,
            Object.fromEntries(
              ast.argNames.props.map(({ prop, alias }) => [alias, prop]),
            ),
          );
        }

        if (
          !Array.isArray(shell.argTypes) &&
          ast.argNames.type === 'identifiers' &&
          ast.argNames.names[0] !== undefined
        ) {
          applyExternals(externalMap, {
            [ast.argNames.names[0]]: Object.fromEntries(
              Object.keys(shell.argTypes).map((arg) => [arg, arg]),
            ),
          });
        }

        // Verifying all required externals are present.
        const missingExternals = ast.externalNames.filter(
          (name) => !(name in externalMap),
        );

        if (missingExternals.length > 0) {
          throw new MissingLinksError(getName(this), missingExternals);
        }

        const args: Snippet[] = Array.isArray(shell.argTypes)
          ? ast.argNames.type === 'identifiers'
            ? shell.argTypes.map((arg, i) => ({
              value: (ast.argNames.type === 'identifiers'
                ? ast.argNames.names[i]
                : undefined) ?? `arg_${i}`,
              dataType: arg as AnyWgslData,
            }))
            : []
          : Object.entries(shell.argTypes).map(([name, dataType]) => ({
            value: name,
            dataType: dataType as AnyWgslData,
          }));

        const { head, body } = ctx.fnToWgsl({
          args,
          returnType: shell.returnType as AnyWgslData,
          body: ast.body,
          externalMap,
        });

        ctx.addDeclaration(
          `${fnAttribute}fn ${id}${ctx.resolve(head)}${ctx.resolve(body)}`,
        );
      }

      return id;
    },
  };

  // The implementation could have been given a name by a bundler plugin,
  // so we try to transfer it to the core.
  const maybeName = getName(implementation);
  if (maybeName !== undefined) {
    setName(core, maybeName);
  }

  return core;
}
