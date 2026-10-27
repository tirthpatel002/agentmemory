// src/errors.ts

export class MemoryError extends Error {
  constructor(message: string) {
    super(message);
    this.name = this.constructor.name;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/** ID unknown in this scope */
export class NotFoundError extends MemoryError {}

/** record() or endRun() called on an ended run */
export class RunClosedError extends MemoryError {}

/** Fail outcome without failureType, bad scope, or invalid event payload */
export class InvalidEventError extends MemoryError {}

/** Invalid state transition (e.g. approving a deleted lesson) */
export class InvalidTransitionError extends MemoryError {}
